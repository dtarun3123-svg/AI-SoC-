import os
import gc
import glob

os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO
from llama_cpp import Llama

# 1. Load YOLOv8 Surface Defect Model (.pte)
print("Loading YOLOv8 Surface Defect Model (.pte) on Pi 4B...")
yolo_model = YOLO("/home/pi/ps5_project/best_executorch_model")

# 2. Load Quantized Llama 3.2 1B Assistant (GGUF - Cortex-A72 Safe)
print("Loading Llama 3.2 1B Quality-Control Assistant...")
llm = Llama(
    model_path="/home/pi/ps5_project/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=False
)

def get_qc_advice(defect_type, confidence):
    """Generates fast, offline QC advice on Pi 4B CPU"""
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a factory quality-control assistant. Reply in 1 short sentence with defect severity and immediate operator action."
            },
            {
                "role": "user",
                "content": f"Bottle defect detected: {defect_type} ({confidence:.0f}% confidence)."
            }
        ],
        max_tokens=45,
        temperature=0.2
    )
    gc.collect()
    return response["choices"][0]["message"]["content"].strip()

# 3. Run Offline Inspection Loop on Sample Images
test_images = sorted(glob.glob("/home/pi/ps5_project/sample_bottles/*.*"))

if not test_images:
    print("No images found in /home/pi/ps5_project/sample_bottles/")
else:
    for img_path in test_images:
        print("\n" + "=" * 55)
        print(f"Inspecting: {os.path.basename(img_path)}")

        results = yolo_model(img_path, verbose=False)
        r = results[0]
        predicted_class = r.names[r.probs.top1]
        confidence = r.probs.top1conf.item() * 100

        print(f"[YOLOv8 Vision] Result: {predicted_class.upper()} ({confidence:.1f}%)")

        if predicted_class == "good":
            print("[QC Status] PASS - Bottle approved for packaging.")
        else:
            print("[QC Status] DEFECT DETECTED - Consulting Llama QC Assistant...")
            advice = get_qc_advice(predicted_class, confidence)
            print(f"[Llama QC Advice] {advice}")