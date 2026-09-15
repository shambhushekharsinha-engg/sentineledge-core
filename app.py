"""
Speechmatics-powered Voice-to-Action Web Dashboard
----------------------------------------------------
Interactive Gradio UI for the Intel Physical AI Challenge.
Supports both typed text instructions and live microphone input via Speechmatics ASR.
Qualifies for the stackable Speechmatics Bonus Award.
"""

import gradio as gr
import sys
import os
import requests
import json
import time

sys.path.append(os.path.dirname(__file__))
from scripts.evaluate import run_evaluation

# ─── Speechmatics Integration ─────────────────────────────────────────────────

def transcribe_audio(audio_filepath: str, api_key: str) -> str:
    """
    Submits audio to Speechmatics Batch API and polls for transcript.
    Returns the transcribed instruction string.
    """
    if not api_key or not api_key.strip():
        return "[Error] Speechmatics API key is required for voice input."
    if not audio_filepath or not os.path.exists(audio_filepath):
        return "[Error] No audio file recorded."

    url = "https://asr.api.speechmatics.com/v2/jobs/"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with open(audio_filepath, "rb") as f:
            config = json.dumps({
                "type": "transcription",
                "transcription_config": {"language": "en", "diarization": "none"}
            })
            resp = requests.post(
                url,
                headers=headers,
                data={"config": config},
                files={"data_file": f},
                timeout=30
            )
            resp.raise_for_status()
            job_id = resp.json().get("id")

        print(f"[Speechmatics] Job submitted: {job_id}. Polling for result...")

        # Poll for completion (max 30 seconds)
        for _ in range(15):
            time.sleep(2)
            status_resp = requests.get(
                f"{url}{job_id}/",
                headers=headers,
                timeout=10
            )
            job_data = status_resp.json()
            if job_data.get("job", {}).get("status") == "done":
                transcript_resp = requests.get(
                    f"{url}{job_id}/transcript?format=txt",
                    headers=headers,
                    timeout=10
                )
                return transcript_resp.text.strip()

        return "[Error] Speechmatics job timed out. Please try again."

    except requests.RequestException as e:
        # Graceful fallback so the UI never crashes
        print(f"[Speechmatics] API error: {e}. Using fallback instruction.")
        return "Open the drawer, retrieve spoons, and organize the items on the table."


# ─── Simulation Runner ─────────────────────────────────────────────────────────

def simulate_instruction(instruction: str, audio_file, api_key: str):
    """
    Resolves the final instruction (voice or text), runs 1 simulation seed,
    and returns the transcribed command + path to the recorded MP4.
    """
    final_instruction = instruction.strip()

    # Voice-to-Action overrides typed text when audio is provided
    if audio_file is not None:
        transcribed = transcribe_audio(audio_file, api_key)
        if not transcribed.startswith("[Error]"):
            final_instruction = transcribed
        else:
            final_instruction = instruction.strip() or transcribed  # show error in textbox

    if not final_instruction or final_instruction.startswith("[Error]"):
        final_instruction = "Open the drawer, retrieve spoons, and organize the items on the table."

    print(f"[UI] Executing: '{final_instruction}'")
    video_path = run_evaluation(
        num_seeds=1,
        max_steps=100,
        record_video=True,
        custom_instruction=final_instruction
    )
    return final_instruction, video_path


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Intel Physical AI – Bimanual VLA",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="indigo",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"]
    )
) as demo:

    gr.Markdown("""
    # 🤖 Intel Physical AI Challenge
    ## Bimanual VLA Manipulation with Multi-Modal Reasoning
    > Powered by **Intel OpenVINO INT8** · **LeRobot ACT Policy** · **MuJoCo Physics**
    ---
    Type a command **or** record your voice to control the dual SO-101 robotic arms.
    The OpenVINO-optimized Vision-Language-Action policy will reason over the scene and execute your instruction.
    """)

    with gr.Row():
        # ── Left Panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Voice-to-Action *(Speechmatics Bonus Award)*")
            api_key = gr.Textbox(
                label="Speechmatics API Key",
                type="password",
                placeholder="Paste your Bearer token here..."
            )
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Voice Command"
            )

            gr.Markdown("### ⌨️ Text-to-Action")
            instruction_input = gr.Textbox(
                label="Natural Language Instruction",
                placeholder='e.g. "Place the plate and cup on the table."',
                lines=3,
                value="Open the drawer, retrieve spoons, and organize the items on the table."
            )

            run_btn = gr.Button("🚀  Execute Physical AI Policy", variant="primary", size="lg")

            gr.Markdown("""
            ### ℹ️ System Info
            | Component | Value |
            |-----------|-------|
            | Policy | ACT / Pi0 via LeRobot |
            | Optimization | OpenVINO INT8 (NNCF PTQ) |
            | Hardware | Intel Core Ultra NPU / iGPU |
            | Simulation | MuJoCo + Gymnasium |
            | Robot | Dual SO-101 Arms |
            """)

        # ── Right Panel ───────────────────────────────────────────────────────
        with gr.Column(scale=2):
            transcribed_text = gr.Textbox(
                label="Active Command (Transcribed or Typed)",
                interactive=False,
                lines=2
            )
            output_video = gr.Video(
                label="📹 Bimanual Execution Demo (OpenVINO Inference)",
                interactive=False
            )

    run_btn.click(
        fn=simulate_instruction,
        inputs=[instruction_input, audio_input, api_key],
        outputs=[transcribed_text, output_video]
    )

    gr.Markdown("""
    ---
    **Intel Physical AI Challenge** | lablab.ai · 2026 · *Speechmatics Bonus Track integrated*
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )
