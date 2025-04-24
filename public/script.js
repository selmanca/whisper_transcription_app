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
const applyBtn   = document.getElementById('apply-workers');
const stopBtn    = document.getElementById('stop-workers');
const countInput = document.getElementById('worker-count');

function createFileEntry(filename) {
  const entry = document.createElement('div');
  entry.className = 'file-entry';
  entry.innerHTML = `<span class="file-name">${filename}</span><span class="status"> - Queued</span>`;
  outputDiv.appendChild(entry);
  return entry;
}

function updateStatus(entryElem, text, showSpinner = false) {
  const statusElem = entryElem.querySelector('.status');
  if (statusElem) {
    // clear then (re)insert text
    statusElem.textContent = ` - ${text}`;

    // add or remove spinner
    let spin = statusElem.querySelector('.spinner');
    if (showSpinner && !spin){
      spin = document.createElement('span');
      spin.className = 'spinner';
      statusElem.appendChild(spin);
    }
    if (!showSpinner && spin){
      spin.remove();
    }
   }
 }

async function handleFiles(files) {
  // Collect base64 strings and file names
  let base64Array = [];
  let fileNames = [];
  
  // Optionally, clear any previous output or show a merged progress entry
  const entry = createFileEntry("Birleştirilen (" + files.length + " ses dosyası)");
  updateStatus(entry, "Yükleniyor...", true);

  // Loop through each file to get its base64 representation
  for (const file of files) {
    fileNames.push(file.name);
    const base64Data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = err => reject(err);
      reader.readAsDataURL(file);
    });
    // Extract only the base64 part (remove header)
    const base64String = base64Data.split(',')[1];
    base64Array.push(base64String);
  }

  try {
    // Send a single POST request with an array of base64-encoded audio files.
    const response = await fetch(RUN_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input: { audio_base64: base64Array } })
    });

    if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
    const result = await response.json();
    const jobId = result.id;
    updateStatus(entry, "İşliyor...", true);

    let statusResponse, statusData;
    const pollInterval = 2000;
    // Poll for job completion.
    while (true) {
      statusResponse = await fetch(STATUS_URL + jobId);
      statusData = await statusResponse.json();

      if (statusData.status === "COMPLETED") break;
      if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(statusData.status)) {
        throw new Error(`Transcription ${statusData.status}`);
      }
      await new Promise(res => setTimeout(res, pollInterval));
    }

    // Get the base64 DOCX output
    const docxBase64 = statusData.output.result || statusData.output;

    // Create a download link for the merged DOCX file
    const downloadLink = document.createElement('a');
    downloadLink.className = 'download-link';
    downloadLink.textContent = "Word dosyasını indir";
    // Using a generic file name for the merged output
    downloadLink.href =
      "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," +
      docxBase64;
    downloadLink.download = "Rapor.docx";

    updateStatus(entry, "Tamamlandı");
    entry.appendChild(downloadLink);
  } catch (err) {
    console.error(err);
    updateStatus(entry, "Başarısız");
  }
}

async function control(n) {
  const url  = n === 0 ? '/stop-workers' : '/update-workers';
  const opts = {
    method : 'POST',
    headers: { 'Content-Type':'application/json' },
    body   : n === 0 ? undefined : JSON.stringify({ max:n })
  };
  applyBtn.disabled = true;   // prevent double-click during request
  stopBtn.disabled  = true;
  await fetch(url, opts);     // ignore body; fetchStatus will refresh UI
}

  applyBtn
    .addEventListener('click', () => control(
      parseInt(document.getElementById('worker-count').value, 10)
    ));

  stopBtn
    .addEventListener('click', () => control(0));
	
async function fetchStatus() {
    const r = await fetch('/workers-status');
    if (!r.ok) return document.getElementById('worker-status').textContent = 'Hata';
    const { workersMax } = await r.json();          // ← use the right field
    document.getElementById('worker-status').textContent =
      workersMax > 0 ? `🟢 Çalışan cihaz sayısı ${workersMax}`
      : '🔴 Cihazlar kapalı';
    // reflect status in controls
    countInput.value = workersMax;
    applyBtn.disabled = workersMax > 0;  // already running → can’t “start” again
    stopBtn.disabled  = workersMax === 0;    
  }

  // Refresh on load
  window.addEventListener('DOMContentLoaded', fetchStatus);

  // Also refresh right after you change workers
  const origControl = control;
  control = async n => {
    await origControl(n);
    fetchStatus();
  };


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