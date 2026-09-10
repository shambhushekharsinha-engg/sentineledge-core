import gradio as gr
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.evaluate import run_evaluation

def simulate_instruction(instruction):
    """
    Runs the MuJoCo simulation for a single seed with the user's custom instruction.
    Returns the path to the generated demo MP4.
    """
    if not instruction.strip():
        instruction = "Open the drawer, retrieve spoons, and organize the items on the table."
        
    print(f"UI Triggered Simulation: '{instruction}'")
    
    # We run 1 seed for 100 max steps for a quick UI response
    video_path = run_evaluation(num_seeds=1, max_steps=100, record_video=True, custom_instruction=instruction)
    
    return video_path

with gr.Blocks(title="Intel Physical AI Challenge - Bimanual VLA") as demo:
    gr.Markdown("# 🤖 Intel Physical AI Challenge: Bimanual VLA Manipulation")
    gr.Markdown("Type a natural language instruction to simulate the dual SO-101 robotic arms. The OpenVINO-optimized VLA policy will process the instruction and visually output the execution.")
    
    with gr.Row():
        with gr.Column(scale=1):
            instruction_input = gr.Textbox(
                label="Natural Language Instruction",
                placeholder="e.g., Pick up the plate with arm A and place it on the table...",
                lines=3
            )
            run_btn = gr.Button("Execute Simulation 🚀", variant="primary")
            
            gr.Markdown("### Setup Info")
            gr.Markdown("""
            - **Target Environment:** MuJoCo
            - **Edge Engine:** Intel OpenVINO (INT8 Quantized)
            - **Robot:** Dual SO-101 Arms
            """)
            
        with gr.Column(scale=2):
            output_video = gr.Video(label="Bimanual Execution Demo", interactive=False)
            
    run_btn.click(
        fn=simulate_instruction,
        inputs=[instruction_input],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
