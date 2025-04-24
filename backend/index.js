import express from 'express';
import path from 'path';
import cors from 'cors';
import dotenv from 'dotenv';
import bodyParser from 'body-parser';
import fetch from 'node-fetch';
import basicAuth from 'express-basic-auth';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { fileURLToPath } from 'url';

// Load .env variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const ENDPOINT_ID = process.env.ENDPOINT_ID;
const API_KEY = process.env.API_KEY;

// 1) Security headers
app.use(helmet());

// // 2) Rate-limit “entrance” only to prevent brute-force
const entranceLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10,                  // allow only 10 attempts per IP
  standardHeaders: true,
  legacyHeaders: false
});

// 3) Basic-Auth for entrance only ───────────────────────────────
const auth = basicAuth({
  users: { [process.env.AUTH_USER]: process.env.AUTH_PASS },
  challenge: true,
  unauthorizedResponse: req => '🛑 Access denied'
});

// Protect and rate-limit only the “entrance” (root + static files)
app.use(
   '/', 
   entranceLimiter, 
   auth, 
   express.static(path.join(__dirname, '../public'))
 );

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Serve static files from public/
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
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

// Update maxWorkers to any value
app.post('/update-workers', async (req, res) => {
  const { max } = req.body;
  const mutation = `
    mutation UpdateWorkers($id: ID!, $min: Int!, $max: Int!) {
      updateEndpoint(input: { id: $id, minWorkers: 0, maxWorkers: $max }) {
        endpoint { id, minWorkers, maxWorkers }
      }
    }
  `;
  try {
    const rp = await fetch(`https://api.runpod.io/graphql?api_key=${API_KEY}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: mutation, variables: { id: ENDPOINT_ID, max } })
    });
    const result = await rp.json();
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Shortcut to set maxWorkers = 0
app.post('/stop-workers', async (_, res) => {
  // just call update-workers with max=0
  req = { body: { max: 0 } };
  return app._router.handle(req, res, () => {});
});

app.get('/workers-status', async (req, res) => {
  const query = `
    query GetEndpoint($id: ID!) {
      endpoint(id: $id) { id, minWorkers, maxWorkers }
    }
  `;
  try {
    const resp = await fetch(
      `https://api.runpod.io/graphql?api_key=${API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, variables: { id: ENDPOINT_ID } })
      }
    );
    const { data, errors } = await resp.json();
    if (errors) return res.status(500).json({ errors });
    res.json(data.endpoint);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Start the server
app.listen(PORT, () => {
    console.log(`✅ Backend running at http://localhost:${PORT}`);
});
