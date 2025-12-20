# Path: plugins/injector_plugin/plugin.py

import gradio as gr
import os
import hashlib
import json
import urllib.request
import html
from pathlib import Path
from shared.utils.plugins import WAN2GPPlugin

class InjectorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Lora Info"
        self.version = "1.1.0"
        self.description = "Displays selected LoRAs Information in HTML with a refresh button.\n Developed:Kovacs-3d"        

    def setup_ui(self):
        self.request_component("loras_choices")
        self.request_global("get_lora_dir")
        self.request_component("model_family")
        self.request_component("model_choice")
        self.request_global("get_model_family")     

    def post_ui_setup(self, components: dict):
        # Save the requested components into instance variables for later use
        self.loras_choices = components.get("loras_choices")
        self.request_global("get_lora_dir")
        self.request_global("get_model_family")
        self.model_family = components.get("model_family")
        self.model_choice = components.get("model_choice")

        def create_inserted_component():
            # Initialize lists to hold gallery data and image information
            gallery_data = []
            image_info = []   

            # Define custom CSS for the gallery container to set a fixed height
            css = """
                .gallery-container { height:600px; }
            """            
            
            # Create an Accordion panel for the LoRA Info plugin, initially closed
            with gr.Accordion("LoRA Info (Plugin)", open=False) as panel:
                # Apply the custom CSS to the panel
                gr.HTML(value=f"<style>{css}</style>")
                
                # Create a row for the refresh and active LoRA buttons
                with gr.Row():
                    refresh_btn = gr.Button("Update Lora List")
                    active_lora_btn = gr.Button("Show only Activated Loras")
                
                # Create an HTML component to display selected image information
                info = gr.HTML(label="Kiválasztott kép információja") 
                
                # Create a Gallery component to display LoRA images in a grid layout
                gallery = gr.Gallery(
                    value=gallery_data,
                    allow_preview=False,
                    object_fit="cover",
                    height="800px",
                    columns=4,
                    rows=2
                )               
                
                def refresh_panel(model_choice, selected_loras=None):
                    def generate_hash(file_path):
                        hash_sha256 = hashlib.sha256()
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_sha256.update(chunk)
                        return hash_sha256.hexdigest()

                    def fetch_civitai_json(hash_value):
                        try:
                            url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
                            response = urllib.request.urlopen(url)
                            data = response.read().decode('utf-8')
                            return json.loads(data)
                        except Exception as e:
                            print(f"[ERROR] Failed to fetch CivitAI JSON data: {e}")
                            return None

                    def create_empty_json(file_path):
                        try:
                            # Derive the JSON file path by replacing the extension
                            json_file_path = file_path.replace('.safetensors', '.json')
                            
                            # Extract the file name without extension for the model name
                            file_name = os.path.basename(file_path).replace('.safetensors', '')
                            
                            # Define the default JSON structure
                            json_data = {
                                "trainedWords": [],
                                "description": "",
                                "model": {
                                    "name": file_name  # Use the file name as the model name
                                },
                                "files": [
                                    {
                                        "hashes": {
                                            "AutoV1": "",
                                            "AutoV2": "",
                                            "SHA256": "",
                                            "CRC32": "",
                                            "BLAKE3": "",
                                            "AutoV3": ""
                                        },
                                        "downloadUrl": ""
                                    }
                                ],
                                "images": [
                                    {
                                        "url": "https://placehold.co/400x300/ff6b6b/white"
                                    }
                                ],
                                "downloadUrl": ""
                            }

                            # Write the default JSON to the file
                            with open(json_file_path, 'w') as f:
                                json.dump(json_data, f, indent=4)
                            
                            print(f"[INFO] Created empty JSON file: {json_file_path}")
                            return json_file_path
                        except Exception as e:
                            print(f"[ERROR] Failed to create empty JSON file: {e}")
                            return None

                    def update_gallery_data(lora_files_dict):
                        # Clear previous data
                        gallery_data.clear()
                        image_info.clear()
                        
                        for safetensors_path, json_path in lora_files_dict.items():
                            try:
                                with open(json_path, 'r') as f:
                                    json_data = json.load(f)
                                
                                # Extract the first image URL or use a placeholder
                                image_url = None
                                if 'images' in json_data and len(json_data['images']) > 0:
                                    image_url = json_data['images'][0]['url']
                                else:
                                    image_url = "https://placehold.co/400x300/cccccc/white?text=Nincs+Kép"
                                
                                # Extract the model name or use a default
                                model_name = "Ismeretlen"
                                if 'model' in json_data and 'name' in json_data['model']:
                                    model_name = json_data['model']['name']
                                
                                gallery_data.append((image_url, model_name))
                                image_info.append(json_path)
                                
                            except Exception as e:
                                print(f"[ERROR] Error processing file: {e}")
                                gallery_data.append(("https://placehold.co/400x300/ff6b6b/white?text=Hiba", "Hibás fájl")) 

                    def collect_lora_files(lora_dir):
                        result_dict = {}
                        
                        if not os.path.exists(lora_dir):
                            print(f"[WARNING] The specified directory does not exist: {lora_dir}")
                            return result_dict
                        
                        # Iterate over files in the directory
                        for filename in os.listdir(lora_dir):
                            if filename.endswith('.safetensors'):
                                file_path = os.path.join(lora_dir, filename)
                                
                                # Derive the JSON file path
                                json_file_path = file_path.replace('.safetensors', '.json')
                                
                                if not os.path.exists(json_file_path):
                                    print(f"[INFO] No JSON file found: {json_file_path}")
                                    
                                    # Generate hash for the .safetensors file
                                    file_hash = generate_hash(file_path)
                                    print(f"[INFO] Generated hash: {file_hash}")
                                    
                                    # Attempt to fetch data from CivitAI
                                    civitai_data = fetch_civitai_json(file_hash)
                                    
                                    if civitai_data:
                                        # Save the fetched data to JSON
                                        with open(json_file_path, 'w') as f:
                                            json.dump(civitai_data, f, indent=2)
                                        print(f"[INFO] Saved CivitAI data: {json_file_path}")
                                    else:
                                        # Create an empty JSON if fetch fails
                                        empty_json_path = create_empty_json(file_path)
                                        if empty_json_path:
                                            json_file_path = empty_json_path
                                else:
                                    print(f"[INFO] JSON file already exists: {json_file_path}")
                                
                                result_dict[file_path] = json_file_path
                        
                        return result_dict  
                        
                    # Retrieve the LoRA directory based on the model choice, default to "None" on error
                    try:
                        lora_dir = self.get_lora_dir(model_choice)
                    except Exception:
                        lora_dir = "None"
                                           
                    print(f"[INFO] Lora dir: {lora_dir}") 

                    # Collect LoRA files and their JSON metadata
                    lora_files_dict = collect_lora_files(lora_dir)
                    
                    # Filter the dictionary if selected LoRAs are provided
                    if selected_loras is not None:  
                        lora_files_dict = {
                            key: value 
                            for key, value in lora_files_dict.items() 
                            if any(lora in key for lora in selected_loras)
                        }                    
                    
                    print(f"[INFO] Number of files found: {len(lora_files_dict)}") 
                    print(f"[INFO] Content: {lora_files_dict}") 

                    # Update the gallery with the collected data
                    update_gallery_data(lora_files_dict)                                       

                    return gr.update(value=gallery_data, selected_index=None), gr.update(value="")
                    
                        
                def show_info(evt: gr.SelectData):
                    if evt.index is not None:
                        json_path = image_info[evt.index]
                        try:
                            # Load JSON data from the file
                            with open(json_path, 'r') as f:
                                json_data = json.load(f)
                            
                            # Extract fields with defaults
                            model_name = "Unknown"
                            model_file_name = "Unknown"
                            description = "No description"
                            download_url = ""
                            trained_words = []
                            
                            # Derive model file name from JSON path
                            model_file_name = os.path.basename(json_path).replace('.json', '')
                            
                            if 'model' in json_data and 'name' in json_data['model']:
                                model_name = json_data['model']['name']
                            
                            if 'description' in json_data:
                                description = json_data['description']
                            
                            if 'modelId' in json_data:
                                download_url = f"https://civitai.com/models/{json_data['modelId']}"   

                            if 'trainedWords' in json_data:
                                trained_words = json_data['trainedWords']                             

                            # Generate HTML for trained words with copy functionality
                            trained_html = ""
                            if trained_words:
                                trained_html = "<div class='trained-word' style='margin-top:10px;'><strong>Trained words:</strong></div>"
                                for word in trained_words:
                                    trained_html += f"""
                                    <h3 onclick="navigator.clipboard.writeText('{html.escape(word)}'); this.style.backgroundColor='#847bb0'; setTimeout(()=>this.style.backgroundColor='#28223d',150);"
                                    style="cursor:pointer; margin-left:10px; float:left; padding:5px; background-color:#28223d; color:#a990f1;">
                                    {html.escape(word)}</h3>
                                    """                                
                                
                            # Construct the full HTML content for display
                            html_content = f"""
                            <div>
                                <h3 style="margin-top: 0;">{model_name}</h3>
                                <p><strong>File name:</strong></p>
                                <h3 onclick="navigator.clipboard.writeText('{html.escape(model_file_name)}'); this.style.backgroundColor='#847bb0'; setTimeout(()=>this.style.backgroundColor='#28223d',150);"
                                style="cursor:pointer; display:inline-block; padding:5px; background-color:#28223d; color:#a990f1; margin-left:10px;">
                                {html.escape(model_file_name)}</h3>
                                <div style="clear:both;"></div>
                                <p><strong>Description:</strong></p>
                                <p>{description}</p>
                                <p><strong>CivitAI Link:</strong></p>
                                <p><a href="{download_url}" target="_blank" style="color: #0066cc;">{download_url}</a></p>
                                {trained_html}
                            </div>
                            """                               

                            return html_content
                        except Exception as e:
                            print(f"[ERROR] Error reading JSON file: {e}")
                            return "An error occurred while loading data."
                    return "No image selected"
                

                # Attach select event to the gallery to show info on selection
                gallery.select(
                    fn=show_info,
                    inputs=None,
                    outputs=info
                )
                
                # Attach click event to the refresh button to update the panel
                refresh_btn.click(
                    fn=refresh_panel,
                    inputs=[self.model_choice],
                    outputs=[gallery, info]
                )

                # Attach click event to the active LoRA button, passing selected LoRAs
                active_lora_btn.click(
                    fn=refresh_panel,
                    inputs=[self.model_choice, self.loras_choices],  
                    outputs=[gallery, info]
                )                

            return panel

        # Insert the created component after the 'queue_accordion' in the UI
        self.insert_after(
            target_component_id="queue_accordion",
            new_component_constructor=create_inserted_component
        )