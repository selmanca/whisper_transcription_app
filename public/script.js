// if we’re on GitHub Pages, use your Render URL;
// otherwise (local‐dev or if you ever host front+back together) use the page’s origin
const API_BASE = window.location.hostname.endsWith('github.io')
  ? 'https://whisper-transcription-app.onrender.com'
  : window.location.origin;

const RUN_URL = `${API_BASE}/transcribe`;
const STATUS_URL = `${API_BASE}/status/`;

const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const outputDiv = document.getElementById('output');

function createFileEntry(filename) {
  const entry = document.createElement('div');
  entry.className = 'file-entry';
  entry.innerHTML = `<span class="file-name">${filename}</span><span class="status"> - Queued</span>`;
  outputDiv.appendChild(entry);
  return entry;
}

function updateStatus(entryElem, text) {
  const statusElem = entryElem.querySelector('.status');
  if (statusElem) statusElem.textContent = " - " + text;
}

async function handleFiles(files) {
  for (const file of files) {
    const entry = createFileEntry(file.name);
    updateStatus(entry, "Uploading...");

    const base64Data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = err => reject(err);
      reader.readAsDataURL(file);
    });

    const base64String = base64Data.split(',')[1];

    try {
      const response = await fetch(RUN_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: { audio_base64: base64String } })
      });

      if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
      const result = await response.json();
      const jobId = result.id;
      updateStatus(entry, "Processing...");

      let statusResponse, statusData;
      const pollInterval = 2000;

      while (true) {
        statusResponse = await fetch(STATUS_URL + jobId);
        statusData = await statusResponse.json();

        if (statusData.status === "COMPLETED") break;
        if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(statusData.status)) {
          throw new Error(`Transcription ${statusData.status}`);
        }

        await new Promise(res => setTimeout(res, pollInterval));
      }

      const docxBase64 = statusData.output.result || statusData.output;

      const downloadLink = document.createElement('a');
      downloadLink.className = 'download-link';
      downloadLink.textContent = "Download Transcript";
      downloadLink.href = "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," + docxBase64;
      downloadLink.download = file.name.replace(/\.[^/.]+$/, "") + "_transcript.docx";

      updateStatus(entry, "Completed");
      entry.appendChild(downloadLink);

    } catch (err) {
      console.error(err);
      updateStatus(entry, "Failed");
    }
  }
}

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
  if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files) handleFiles(fileInput.files);
});
