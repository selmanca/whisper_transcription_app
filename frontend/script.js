// **Configuration**: Set your RunPod Endpoint ID and API key here.
const ENDPOINT_ID = "<YOUR_ENDPOINT_ID>";   // e.g., "abcd1234... (as provided by RunPod)"
const API_KEY = "<YOUR_RUNPOD_API_KEY>";    // Your RunPod API key (keep this secret!)

// URLs for RunPod endpoint operations
const RUN_URL = `https://api.runpod.ai/v2/${ENDPOINT_ID}/run`;
const STATUS_URL = `https://api.runpod.ai/v2/${ENDPOINT_ID}/status/`;

// Get page elements
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const outputDiv = document.getElementById('output');

// Utility: Create a UI entry for a file
function createFileEntry(filename) {
  const entry = document.createElement('div');
  entry.className = 'file-entry';
  entry.innerHTML = `<span class="file-name">${filename}</span><span class="status"> - Queued</span>`;
  outputDiv.appendChild(entry);
  return entry;
}

// Utility: Update status text for a file entry
function updateStatus(entryElem, text) {
  const statusElem = entryElem.querySelector('.status');
  if (statusElem) statusElem.textContent = " - " + text;
}

// Handle file uploads
async function handleFiles(files) {
  for (const file of files) {
    // Create UI entry and show "Uploading..." status
    const entry = createFileEntry(file.name);
    updateStatus(entry, "Uploading...");

    // Read the file as base64 data URL
    const base64Data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = err => reject(err);
      reader.readAsDataURL(file);
    });

    // Extract the base64 string (remove the data URL prefix)
    const base64String = base64Data.split(',')[1];

    try {
      // Send the transcription request to RunPod (asynchronous job)
      const response = await fetch(RUN_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': API_KEY  // RunPod uses the API key as auth header
        },
        body: JSON.stringify({ input: { audio_base64: base64String } })
      });
      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`);
      }
      const result = await response.json();
      const jobId = result.id;
      updateStatus(entry, "Processing...");

      // Poll the status until the job is completed
      let statusResponse, statusData;
      const pollInterval = 2000;  // 2 seconds
      while (true) {
        statusResponse = await fetch(STATUS_URL + jobId, {
          method: 'GET',
          headers: { 'Authorization': API_KEY }
        });
        statusData = await statusResponse.json();
        if (statusData.status === "COMPLETED") {
          break;
        } else if (statusData.status === "FAILED" || statusData.status === "CANCELLED" || statusData.status === "TIMED_OUT") {
          throw new Error(`Transcription ${statusData.status}`);
        }
        // Wait then poll again
        await new Promise(res => setTimeout(res, pollInterval));
      }

      // Job completed, get the transcription result (base64 docx content)
      const output = statusData.output;
      if (!output) {
        throw new Error("No output from transcription");
      }

      // The handler returns a JSON with the base64 string (we used key "result")
      const docxBase64 = output.result || output;  // support if returned as raw string

      // Create a download link for the .docx file
      const downloadLink = document.createElement('a');
      downloadLink.className = 'download-link';
      downloadLink.textContent = "Download Transcript";
      downloadLink.href = "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," + docxBase64;
      // Set download filename (original name without extension + "_transcript.docx")
      const baseName = file.name.replace(/\.[^/.]+$/, "");
      downloadLink.download = baseName + "_transcript.docx";

      // Update UI: mark as completed and append the download link
      updateStatus(entry, "Completed");
      entry.appendChild(downloadLink);
    } catch (err) {
      console.error(err);
      updateStatus(entry, "Failed");
    }
  }
}

// Drag-and-drop handlers for user convenience
uploadSection.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadSection.classList.add('dragover');
});
uploadSection.addEventListener('dragleave', () => {
  uploadSection.classList.remove('dragover');
});
uploadSection.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadSection.classList.remove('dragover');
  if (e.dataTransfer.files) {
    handleFiles(e.dataTransfer.files);
  }
});

// Handle manual file selection via input
fileInput.addEventListener('change', () => {
  if (fileInput.files) {
    handleFiles(fileInput.files);
  }
});
