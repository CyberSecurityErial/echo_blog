import ast
import re
import unittest
from pathlib import Path


POST_PATH = Path(
    "content/posts/llm-system/training-schedule/"
    "llm-system-schedule-01-training-framework-schedule.md"
)


def read_post():
    return POST_PATH.read_text(encoding="utf-8")


def split_front_matter(text):
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    if not match:
        raise AssertionError("post must use YAML front matter delimited by ---")
    return match.group(1), match.group(2)


def parse_simple_front_matter(front_matter):
    metadata = {}
    for line in front_matter.splitlines():
        key, value = line.split(": ", 1)
        value = value.strip()
        if value in {"true", "false"}:
            metadata[key] = value == "true"
        elif value.startswith("["):
            metadata[key] = ast.literal_eval(value)
        elif value.startswith("'") and value.endswith("'"):
            metadata[key] = value[1:-1]
        elif value.isdigit():
            metadata[key] = int(value)
        else:
            metadata[key] = value
    return metadata


class LLMSystemSchedulePostTest(unittest.TestCase):
    def setUp(self):
        front_matter, body = split_front_matter(read_post())
        self.metadata = parse_simple_front_matter(front_matter)
        self.body = body

    def test_front_matter_matches_series_conventions(self):
        self.assertEqual(
            self.metadata["title"],
            "LLM System: Training Schedule 01 - 训练框架中的 Schedule 算法",
        )
        self.assertEqual(self.metadata["date"], "2026-06-08T00:00:00+08:00")
        self.assertFalse(self.metadata["draft"])
        self.assertEqual(self.metadata["categories"], ["LLM System"])
        self.assertEqual(self.metadata["series"], ["LLM System", "Training Schedule"])
        self.assertEqual(self.metadata["series_order"], 1)
        self.assertEqual(self.metadata["weight"], 1)
        self.assertTrue(self.metadata["math"])

    def test_tags_cover_training_schedule_topic(self):
        self.assertEqual(
            self.metadata["tags"],
            [
                "LLM",
                "LLM System",
                "Training",
                "Schedule",
                "Pipeline Parallel",
                "Distributed Training",
            ],
        )

    def test_body_keeps_current_outline_order(self):
        expected_headings = [
            "## 问题背景",
            "## PP",
            "## GPipe",
            "## 1F1B",
            "## interleaved 1F1B",
            "## Chimera",
            "## zero-bubble",
            "## DualPipe",
            "## DualPipeV",
            "## 为什么不考虑通信？",
            "## 削峰填谷",
            "## moe有何不同",
            "## 如何实现一个调度器",
        ]
        positions = [self.body.index(heading) for heading in expected_headings]

        self.assertIn("> 本篇目标：", self.body)
        self.assertEqual(positions, sorted(positions))

    def test_perfetto_links_are_inserted(self):
        expected_trace_files = [
            "gpipe_trace.json",
            "1f1b_trace.json",
            "chimera_trace.json",
            "zerobubble_1f1b_trace.json",
            "dualpipe_trace.json",
            "dualpipev_trace.json",
            "moe_bad_overlap_1f1b_trace.json",
        ]

        for trace_file in expected_trace_files:
            with self.subTest(trace_file=trace_file):
                self.assertIn(
                    "https://ui.perfetto.dev/#!/?url="
                    "https://CyberSecurityErial.github.io/echo_blog/traces/"
                    f"{trace_file}",
                    self.body,
                )

    def test_gradient_equations_are_math_blocks(self):
        self.assertIn("$$\nY = XW\n$$", self.body)
        self.assertIn("$$\ndX = dY W^T,\\quad dW = X^T dY\n$$", self.body)


if __name__ == "__main__":
    unittest.main()
