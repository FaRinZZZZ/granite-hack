
---

# **Granite-chan: Advanced RAG Supercenter Assistant**

Welcome to the **Granite-chan** project, an advanced **retrieval-augmented generation (RAG)** system designed to help customers navigate **Granite Supercenter** with both an **online AI chatbot** and an **offline physical robot**.

Granite-chan offers **personalized shopping assistance** using a **tsundere** personality, integrated with **IBM WatsonX** for natural language understanding and **Piper TTS** for voice generation.

This repository contains:
- The **AI chatbot** for online use.
- The **offline robot assistant** for guiding customers in the physical store.
- **Demo video** showcasing the capabilities.

---

## **Project Structure**
The project is organized as follows:

```
/Granite-chan
    /data                     # Contains the PDF files with supercenter details
    /piper                     # Contains the Piper TTS model and config files
    requirements.txt          # Python dependencies required for the project
    advanceRAG_api.py          # The core API logic for response generation using RAG
    main.py                    # Main file to run the application and test Granite-chan
    demo_video.mp4             # Demo video showcasing the robot and AI chatbot in action
```

---

## **1. Project Setup**

Before running the project, you need to install the required Python dependencies. This is crucial to ensure that all libraries and frameworks work smoothly.

### **1.1 Install Dependencies**

1. **Create a Virtual Environment** (optional but recommended):

   ```bash
   conda create --name granite-chan python=3.11 -y
   conda activate granite-chan
   ```

2. **Install Required Libraries** using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary packages for running **Granite-chan**:
   - `langchain`
   - `FAISS`
   - `ibm-watsonx-ai`
   - `Piper TTS`
   - `RealtimeSTT`
   - Other dependencies for document loading and model inference

### **1.2 Setup the Files**
Make sure the project directory is structured like this:

```
/Granite-chan
    /data                     # Your PDFs with store information (Granite Supercenter Guide)
    /piper                     # Contains Piper TTS model files and configuration
    advanceRAG_api.py          # The main logic for response generation
    main.py                    # Main script to run
    demo_video.mp4             # Demo video
```

### **1.3 Configure Your Paths**
Ensure that the following paths are correctly set in your **main.py** and **advanceRAG_api.py**:

- **Piper model files and configuration**:
  - **`piper/en_GB-southern_english_female-low.onnx`**
  - **`piper/en_GB-southern_english_female-low.onnx.json`**

- **PDF Files**:
  - Store your **Granite Supercenter Detailed Guide PDF** in the `/data` folder.

---

## **2. Running the Application**

### **2.1 Run in Admin Mode**
Because of the **Piper TTS** engine and its access to system resources, it’s recommended to **run the application as Administrator** to avoid permission issues.

1. **Open VS Code as Administrator**:
   - Right-click on **VS Code** and select **Run as Administrator**.

2. **Run the `main.py` file**:
   - Make sure you're in the **project directory** and run:
   
   ```bash
   python main.py
   ```

   The application will:
   - Start listening to speech input.
   - Process the speech-to-text output.
   - Generate **Granite-chan's response** and speak it aloud via **Piper TTS**.
   - The **destination** (store zone) will also be displayed in the terminal.

---

## **3. Functionality Walkthrough**

### **3.1 Chatbot Interaction (Online)**

- **Granite-chan** acts as a **tsundere assistant** for **online shopping**. Users can interact with the chatbot to ask questions about store sections, products, promotions, etc.
- **Granite-chan** responds with a **playful and sarcastic tone**, guiding users through the store’s available sections and promotions.
  
  Example:
  ```
  User: Where can I buy a new bed?
  Granite-chan: Ugh, you want to buy a bed? Fine, I’ll take you to the furniture section... but don’t think I care, okay?
  Destination: furniture
  ```

### **3.2 Physical Robot Interaction (Offline)**

- **Granite-chan’s physical robot** uses **Piper TTS** to give directions and guide customers in the **supercenter**.
- The robot helps **guide customers** to different **zones** (like **Furniture**, **Food**, **Restrooms**, etc.) while **maintaining her tsundere personality**.

### **3.3 RAG-Based Responses**

- The system uses **Retrieval-Augmented Generation (RAG)** to provide **contextual answers** by **retrieving relevant data** from the supercenter documents (PDF/TXT).
- The **Granite Assistant** dynamically answers based on **available context** and **customer queries**.

Example:
```
User: What are the promotions in the clothing section?
Granite-chan: Oh, you want the promotions? Fine! Buy 1, Get 1 Free on T-shirts for Men and Women, and 50% Off the Summer Collection. Happy now?
Destination: clothing
```

---

## **4. Demo Video**

To view the demo video, simply open the file:

```
/demo_video.mp4

```
https://github.com/user-attachments/assets/68c81b1a-9e73-4577-9876-7f7a189b8d37

This video showcases:
- **Granite-chan’s interaction** in both the **online chatbot** and **offline robot** versions.
- **Granite-chan’s tsundere personality** as she guides the user around the **Granite Supercenter**.

---

## **5. License & Acknowledgements**

This project is licensed under the **MIT License**. See the **LICENSE** file for more details.

---

## **6. Troubleshooting**

- **Error: "PermissionError: [WinError 5] Access is denied"**:
  - Make sure you **run VS Code as Administrator** or **ensure Piper’s `.exe` file has appropriate permissions**.
  
- **Error with `Piper TTS`**:
  - Ensure that the correct **Piper TTS model** is specified in the `main.py` file.
  - Verify the path for the `piper.exe` is correct and **accessible**.

---

### **7. Contributing**

We welcome contributions! If you have any suggestions or fixes, feel free to submit a **pull request**. Please follow the **contribution guidelines** in the repository.

---

### **8. Contact**

For any questions, feel free to contact us via **our github** or create an **issue** in the repository.

---

## **Final Notes**
- **Granite-chan** is designed to improve the **shopping experience**, whether online or in the **supercenter**.
- **Piper TTS** brings a fun and engaging personality to the assistant, adding a **unique tsundere charm**.
- The **Advanced RAG** allows the assistant to **intelligently respond** based on the supercenter’s data, while the **physical robot** can guide customers in real time!

--- 
