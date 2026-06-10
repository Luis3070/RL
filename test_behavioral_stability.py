from __future__ import annotations

import json
import unittest

from behavioral_stability import aggregate_rep_rows, parse_scenario_list, parse_seed_list, split_failures


class BehavioralStabilityTests(unittest.TestCase):
    def test_parse_seed_list_defaults_and_csv(self) -> None:
        self.assertEqual(parse_seed_list(""), (11, 23, 37, 51, 67, 83))
        self.assertEqual(parse_seed_list("4200, 4213,4229"), (4200, 4213, 4229))

    def test_parse_scenario_list_and_failure_split(self) -> None:
        self.assertEqual(
            parse_scenario_list("dynamic_full_field_pressure, blocked_corridor_repair"),
            ("dynamic_full_field_pressure", "blocked_corridor_repair"),
        )
        self.assertEqual(
            split_failures("collision_rate<=0.12; clean_episode_rate>=0.50"),
            ("collision_rate<=0.12", "clean_episode_rate>=0.50"),
        )

    def test_aggregate_rep_rows_reports_pass_rate_and_failure_histogram(self) -> None:
        rows = [
            {
                "passed": 1,
                "failures": "",
                "collision_rate": 0.0,
                "clean_episode_rate": 0.8,
                "corridor_keep_rate": 0.9,
            },
            {
                "passed": 0,
                "failures": "collision_rate<=0.12",
                "collision_rate": 0.25,
                "clean_episode_rate": 0.4,
                "corridor_keep_rate": 0.7,
            },
            {
                "passed": 1,
                "failures": "",
                "collision_rate": 0.1,
                "clean_episode_rate": 0.7,
                "corridor_keep_rate": 0.8,
            },
        ]

        aggregate = aggregate_rep_rows(
            rows,
            pass_rate_threshold=0.66,
            metric_names=("collision_rate", "clean_episode_rate", "corridor_keep_rate"),
        )

        self.assertEqual(aggregate["reps"], 3)
        self.assertEqual(aggregate["pass_count"], 2)
        self.assertAlmostEqual(float(aggregate["pass_rate"]), 2.0 / 3.0)
        self.assertEqual(aggregate["stable_passed"], 1)
        self.assertAlmostEqual(float(aggregate["collision_rate_max"]), 0.25)
        self.assertAlmostEqual(float(aggregate["clean_episode_rate_min"]), 0.4)
        self.assertEqual(json.loads(str(aggregate["failure_counts_json"])), {"collision_rate<=0.12": 1})


if __name__ == "__main__":
    unittest.main()
