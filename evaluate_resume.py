import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from prompts import SYSTEM_PROMPT, build_evaluation_input
from schemas import ResumeEvaluation


PROJECT_ROOT = Path(__file__).resolve().parent


def read_text_file(path: Path) -> str:
    """Read UTF-8 text, including files with a Windows UTF-8 BOM."""
    return path.read_text(encoding="utf-8-sig")


def sanitize_json_response(raw_text: str) -> str:
    """Remove whitespace and an optional whole-response code fence."""
    text = raw_text.strip()

    fenced = re.fullmatch(
        r"```(?:json)?\s*\n(.*?)\n```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        return fenced.group(1).strip()

    return text


def evaluate_resume(
    client: genai.Client,
    model: str,
    resume_text: str,
    job_description: str,
) -> ResumeEvaluation:
    """Generate an evaluation and validate it before returning."""
    document_input = build_evaluation_input(
        resume_text,
        job_description,
    )

    response = client.models.generate_content(
        model=model,
        contents=document_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=ResumeEvaluation.model_json_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),    
        ),
    )

    if not response.candidates:
        raise ValueError(
            "Gemini returned no candidate response. "
            "The request may have been blocked."
        )

    finish_reason = response.candidates[0].finish_reason

    if finish_reason != types.FinishReason.STOP:
        raise ValueError(
            f"Generation did not complete normally: {finish_reason}. "
            "No evaluation was accepted."
        )

    raw_text = response.text

    if not raw_text or not raw_text.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned_text = sanitize_json_response(raw_text)

    return ResumeEvaluation.model_validate_json(cleaned_text)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate resume text against a JD using Gemini."
    )

    parser.add_argument(
        "--resume",
        type=Path,
        required=True,
        help="Path to the resume text file.",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        required=True,
        help="Path to the job description text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the validated JSON output.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env.", file=sys.stderr)
        return 1

    if not model:
        print("ERROR: Set GEMINI_MODEL in .env.", file=sys.stderr)
        return 1

    started = time.perf_counter()

    try:
        # Prevent accidentally overwriting either input file.
        if args.output:
            output_path = args.output.resolve()
            input_paths = {args.resume.resolve(), args.jd.resolve()}

            if output_path in input_paths:
                raise ValueError(
                    "Output path must differ from both input paths."
                )

        resume_text = read_text_file(args.resume)
        job_description = read_text_file(args.jd)

        # Validate inputs before making any network request.
        build_evaluation_input(resume_text, job_description)

        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=2.0,
                    max_delay=10.0,
                    exp_base=2.0,
                    jitter=1.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                ),            
            ),
        ) as client:
            result = evaluate_resume(
                client=client,
                model=model,
                resume_text=resume_text,
                job_description=job_description,
            )

        output_json = result.model_dump_json(indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                output_json + "\n",
                encoding="utf-8",
            )

        # Only validated evaluation JSON goes to stdout.
        print(output_json)

        elapsed = time.perf_counter() - started
        print(
            f"Evaluation completed in {elapsed:.2f}s.",
            file=sys.stderr,
        )

        if args.output:
            print(f"Saved to: {args.output}", file=sys.stderr)

        return 0

    except ValidationError as exc:
        # Exclude rejected values to avoid logging resume/model content.
        details = exc.errors(
            include_input=False,
            include_context=False,
            include_url=False,
        )
        print(
            f"ERROR: Output failed schema validation: {details}",
            file=sys.stderr,
        )

    except errors.APIError as exc:
        message = str(exc.message).replace(api_key, "[REDACTED]")
        print(
            f"ERROR: Gemini API request failed ({exc.code}): {message}",
            file=sys.stderr,
        )

    except httpx.TimeoutException:
        print(
            "ERROR: Gemini request timed out. No evaluation was produced.",
            file=sys.stderr,
        )

    except httpx.TransportError:
        print(
            "ERROR: Network connection failed. No evaluation was produced.",
            file=sys.stderr,
        )

    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())