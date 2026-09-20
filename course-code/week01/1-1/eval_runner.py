import asyncio
import json
from pathlib import Path
from agent_loop_demo import agent_loop


def load_cases() -> list[dict]:
    cases_path = Path(__file__).with_name("eval_cases.json")
    return json.loads(cases_path.read_text(encoding="utf-8"))


def find_matching_tool_call(tool_calls: list[dict], tool_name: str) -> dict | None:
    for call in tool_calls:
        if call["name"] == tool_name:
            return call
    return None


def arguments_match_exact(call: dict, expected_arguments: dict) -> bool:
    actual_arguments = call.get("arguments", {})
    return all(actual_arguments.get(key) == value for key, value in expected_arguments.items())


def arguments_match_contains(call: dict, expected_arguments_contains: dict) -> bool:
    actual_arguments = call.get("arguments", {})
    for key, tokens in expected_arguments_contains.items():
        actual_value = str(actual_arguments.get(key, ""))
        if not all(token in actual_value for token in tokens):
            return False
    return True


async def main2() -> None:
    cases = load_cases()
    passed = 0
    total = len(cases)

    for idx, case in enumerate(cases, start=1):
        result = await agent_loop(case["input"], verbose=False)
        tool_calls = result["tool_calls"]
        used_tools = [call["name"] for call in tool_calls]
        answer = result["final_answer"]

        matched_call = find_matching_tool_call(tool_calls, case["expect_tool"])
        tool_ok = matched_call is not None
        args_ok = tool_ok
        if matched_call and "expect_arguments" in case:
            args_ok = args_ok and arguments_match_exact(
                matched_call, case["expect_arguments"]
            )
        if matched_call and "expect_arguments_contains" in case:
            args_ok = args_ok and arguments_match_contains(
                matched_call, case["expect_arguments_contains"]
            )
        answer_ok = all(word in answer for word in case["expect_answer_contains"])
        ok = tool_ok and args_ok and answer_ok
        if ok:
            passed += 1

        print(
            f"[case {idx}] ok={ok} "
            f"tool_ok={tool_ok} args_ok={args_ok} answer_ok={answer_ok} "
            f"tools={used_tools} duration_ms={result['duration_ms']}"
        )
        print(f"  input: {case['input']}")
        if matched_call:
            print(f"  matched_call: {matched_call}")
        print(f"  answer: {answer}")

    print(f"\nPass rate: {passed}/{total}")


if __name__ == '__main__':
    asyncio.run(main2())
