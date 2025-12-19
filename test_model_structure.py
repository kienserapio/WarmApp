#!/usr/bin/env python3
"""
Step 1: Verify Model Structure
This script checks if your TensorFlow.js models can be converted and used
"""
import json
import yaml
import os

print("=" * 70)
print("🔍 STEP 1: VERIFYING MODEL STRUCTURE")
print("=" * 70)

# Check one model as a representative sample
model_path = "/Users/kien/Files/workout/models/WarmApp_PushUp_Model/model.json"
metadata_path = "/Users/kien/Files/workout/models/WarmApp_PushUp_Model/metadata.yaml"

print(f"\nChecking: {model_path}")
print("-" * 70)

# Check if files exist
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit(1)

if not os.path.exists(metadata_path):
    print(f"❌ Metadata file not found: {metadata_path}")
    exit(1)

print("✓ Files exist")

# Read model.json
try:
    with open(model_path, 'r') as f:
        model_config = json.load(f)
    
    print("\n📊 MODEL ARCHITECTURE:")
    print("-" * 70)
    
    # Check if it's a proper TensorFlow.js model
    if 'modelTopology' in model_config:
        print("✓ Valid TensorFlow.js model format detected")
        
        # Get model topology
        topology = model_config['modelTopology']
        
        # Check for model config
        if 'model_config' in topology:
            config = topology['model_config']
            
            # Get layers
            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
                
                print(f"\n📐 Number of layers: {len(layers)}")
                
                # First layer (input)
                first_layer = layers[0]
                print(f"\n🔹 Input Layer:")
                print(f"   Type: {first_layer.get('class_name', 'Unknown')}")
                
                if 'config' in first_layer:
                    if 'batch_input_shape' in first_layer['config']:
                        input_shape = first_layer['config']['batch_input_shape']
                        print(f"   Shape: {input_shape}")
                        
                        # Verify it's an image input
                        if len(input_shape) == 4 and input_shape[-1] == 3:
                            print(f"   ✓ IMAGE INPUT MODEL (batch, height, width, channels)")
                            print(f"   ✓ Expected input: {input_shape[1]}x{input_shape[2]} RGB images")
                        else:
                            print(f"   ⚠️  Unexpected input shape: {input_shape}")
                
                # Last layer (output)
                last_layer = layers[-1]
                print(f"\n🔹 Output Layer:")
                print(f"   Type: {last_layer.get('class_name', 'Unknown')}")
                
                if 'config' in last_layer:
                    if 'units' in last_layer['config']:
                        num_classes = last_layer['config']['units']
                        print(f"   Number of output classes: {num_classes}")
                    
                    if 'activation' in last_layer['config']:
                        activation = last_layer['config']['activation']
                        print(f"   Activation: {activation}")
                        
                        if activation == 'softmax':
                            print(f"   ✓ CLASSIFICATION MODEL (softmax output)")
            
            # Check for graph model format
            elif 'node' in topology:
                print("\n⚠️  Graph model format detected (more complex)")
                nodes = topology['node']
                print(f"   Number of nodes: {len(nodes)}")
                
                # Try to find input/output info
                if 'signature' in model_config.get('userDefinedMetadata', {}):
                    print(f"   Signature info available")
        
        # Check weights manifest
        if 'weightsManifest' in model_config:
            manifest = model_config['weightsManifest']
            print(f"\n💾 Weights Info:")
            print(f"   Number of weight groups: {len(manifest)}")
            
            total_weights = 0
            for group in manifest:
                if 'weights' in group:
                    total_weights += len(group['weights'])
            print(f"   Total weight tensors: {total_weights}")
            
            # Check weight files
            if manifest and 'paths' in manifest[0]:
                weight_files = manifest[0]['paths']
                print(f"   Weight files: {len(weight_files)} shard(s)")
                
                # Verify weight files exist
                model_dir = os.path.dirname(model_path)
                missing_files = []
                for weight_file in weight_files:
                    weight_file_path = os.path.join(model_dir, weight_file)
                    if not os.path.exists(weight_file_path):
                        missing_files.append(weight_file)
                
                if missing_files:
                    print(f"   ❌ Missing weight files: {missing_files}")
                else:
                    print(f"   ✓ All weight files present")
    else:
        print("❌ Not a valid TensorFlow.js model format")
        exit(1)

except json.JSONDecodeError as e:
    print(f"❌ Error reading model.json: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Read metadata.yaml
print("\n" + "=" * 70)
print("📋 METADATA:")
print("-" * 70)

try:
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    print(f"Description: {metadata.get('description', 'N/A')}")
    print(f"Task: {metadata.get('task', 'N/A')}")
    print(f"Date: {metadata.get('date', 'N/A')}")
    
    # Look for class names
    if 'names' in metadata:
        names = metadata['names']
        print(f"\n🏷️  Class Names Found:")
        if isinstance(names, dict):
            for key, value in names.items():
                print(f"   {key}: {value}")
        elif isinstance(names, list):
            for i, name in enumerate(names):
                print(f"   {i}: {name}")
    
    if 'kpt_shape' in metadata:
        kpt_shape = metadata['kpt_shape']
        print(f"\n🎯 Keypoint Shape: {kpt_shape}")
        print(f"   This is a POSE DETECTION model (outputs keypoints)")
    
    if 'kpt_names' in metadata:
        print(f"   ⚠️  This is a YOLO POSE model, not a classification model")
        print(f"   ⚠️  It outputs keypoint coordinates, not form quality scores")

except yaml.YAMLError as e:
    print(f"❌ Error reading metadata.yaml: {e}")
except Exception as e:
    print(f"⚠️  Could not read metadata: {e}")

# Final assessment
print("\n" + "=" * 70)
print("🎯 ASSESSMENT:")
print("=" * 70)

# Check if this is actually a pose detection model (YOLO) or classification model
with open(metadata_path, 'r') as f:
    metadata = yaml.safe_load(f)

if 'kpt_shape' in metadata and 'task' in metadata and metadata['task'] == 'pose':
    print("\n⚠️  IMPORTANT DISCOVERY:")
    print("   Your models are YOLO POSE DETECTION models, not classification models!")
    print("\n   What they do:")
    print("   • Detect person bounding boxes")
    print("   • Output 17 keypoint coordinates (x, y, confidence)")
    print("   • Similar to MediaPipe, but trained on your specific exercises")
    print("\n   What they DON'T do:")
    print("   • They don't classify 'good form' vs 'bad form'")
    print("   • They don't output quality scores")
    print("\n   ❌ CANNOT BE USED FOR FORM CLASSIFICATION")
    print("\n   ✅ POSSIBLE SOLUTIONS:")
    print("   1. Use MediaPipe (already better for keypoint detection)")
    print("   2. Retrain models in Teachable Machine for IMAGE CLASSIFICATION")
    print("      (Label images as 'good_form', 'bad_form', etc.)")
    print("   3. Train a custom classifier using MediaPipe keypoints as input")
else:
    print("\n✅ This appears to be a classification model")
    print("   Can proceed with conversion and integration")

print("\n" + "=" * 70)
print("Next step: Run this to see the verdict")
print("=" * 70)
