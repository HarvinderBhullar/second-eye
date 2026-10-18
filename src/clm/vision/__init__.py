"""Second Eye: a vision state tower for CLM.

A frozen image encoder (C-RADIOv2-H, the vision tower of Nemotron Nano 2 VL, or the
full Nemotron VL) feeds a small trainable image *state* head, scored against CLM's
existing frozen *action* tower (Qwen3-8B + the released action head) embedding the
answer options.  See docs/VISION.md.  Research prototype, not a medical device.
"""
QUESTION = "Is there a bone fracture in this X-ray?"
OPTIONS = ["Yes, there is a fracture.", "No, there is no fracture."]   # index 0 = positive class
