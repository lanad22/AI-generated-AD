import unittest

from prepare_final_data import prepare_audio_clips


class PrepareAudioClipsVideoLengthTests(unittest.TestCase):
    def test_overflowing_clip_becomes_extended(self):
        clips = [
            {
                "scene_number": 1,
                "text": "A long explanation",
                "type": "Visual",
                "start_time": 8.0,
                "tts_duration": 10.0,
            }
        ]

        prepared = prepare_audio_clips(1, clips, [], video_length=10.0)

        self.assertEqual(prepared[0]["track_type"], "extended")
        self.assertEqual(prepared[0]["end_time"], 8.0)

    def test_in_bounds_clip_stays_inline(self):
        clips = [
            {
                "scene_number": 1,
                "text": "A short explanation",
                "type": "Visual",
                "start_time": 3.0,
                "end_time": 5.0,
            }
        ]

        prepared = prepare_audio_clips(1, clips, [], video_length=10.0)

        self.assertEqual(prepared[0]["track_type"], "inline")
        self.assertEqual(prepared[0]["end_time"], 5.0)


if __name__ == "__main__":
    unittest.main()
