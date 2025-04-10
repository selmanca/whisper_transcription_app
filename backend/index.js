import express from 'express';
import path from 'path';
import cors from 'cors';
import dotenv from 'dotenv';
import bodyParser from 'body-parser';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';

// Setup directory constants
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const ENDPOINT_ID = process.env.ENDPOINT_ID;
const API_KEY = process.env.API_KEY;

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Serve static files from public/
app.use(express.static(path.join(__dirname, '../public')));

// Transcription request handler
app.post('/transcribe', async (req, res) => {
    const audio_base64 = req.body.input?.audio_base64;
    if (!audio_base64) {
        return res.status(400).json({ error: 'Missing audio_base64' });
    }

    try {
        const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + process.env.API_KEY
            },
            body: JSON.stringify({ input: { audio_base64 } })
        });

        if (!response.ok) {
            const errorText = await response.text();
            return res.status(response.status).json({ error: errorText });
        }

        const data = await response.json();
        res.json(data); // contains job ID
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Status polling handler
app.get('/status/:id', async (req, res) => {
    const jobId = req.params.id;

    try {
        const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${jobId}`, {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer ' + process.env.API_KEY
            }
        });

        if (!response.ok) {
            const errorText = await response.text();
            return res.status(response.status).json({ error: errorText });
        }

        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`✅ Backend running at http://localhost:${PORT}`);
});
