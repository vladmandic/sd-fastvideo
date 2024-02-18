import { WebCam } from "./webcam.js";
import { initNVML } from "./nvml.js";

const clientID = Date.now();
const debounceTime = 250;
const timers = {};
let ws = null;
let data = null;
window.options = {};
let webcam = null;
let image = null;
let images = [];
const receiveTimes = [0];

const wait = (ms) => new Promise((res) => setTimeout(res, ms));

window.log = (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (msg) console.log(ts, ...msg); // eslint-disable-line no-console
}

async function sendImage() {
  if (data.readyState !== 1) await wait(250);
  if (!webcam) await wait(250)
  if (webcam.paused) return;
  if (!data) initData()
  const jpeg = webcam.snap()
  const encoded = new TextEncoder().encode(jpeg);
  if (data.readyState === 1) data.send(encoded)
}

async function initWS() {
  ws = new WebSocket(`ws://localhost:8000/ws/${clientID}`);
  log('ws init', ws)
  ws.onmessage = (event) => {
    const json = JSON.parse(event.data)
    // log('ws receive', json)
    if (json.ready) sendImage() // send new image on ready
    else {
      for (const [id, data] of Object.entries(json)) {
        const el = document.getElementById(id)
        if (el) el.value = data // update bound dom elements
        else options[id] = data // update options
      }
    }
  };
  setInterval(() => ws.readyState === 1 ? ws.send(JSON.stringify({ ready: true })) : {}, 100); // poll for ready
}

async function initData() {
  data = new WebSocket(`ws://localhost:8000/data/${clientID}`);
  log('ws data', data)
  let receiveTime = Date.now();
  data.onmessage = (event) => {
    // log('data receive', event.data)
    receiveTimes.push(Date.now() - receiveTime);
    receiveTime = Date.now();
    if (receiveTimes.length > 100) receiveTimes.shift();
    if (!image) {
      image = new Image();
      image.onload = () => images.push(image);
    }
    try {
      image.src = URL.createObjectURL(event.data);
    } catch {}
  };
}

async function drawImage() {
  const receiveAvgDelay = receiveTimes.reduce((a, b) => a + b, 100) / (receiveTimes.length);
  setTimeout(drawImage, receiveAvgDelay);
  if (images.length === 0) return;
  image = images.shift();
  const canvas = document.getElementById('output');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  document.getElementById('fps').innerText = `FPS: ${(1000 / receiveAvgDelay).toFixed(2)}`;
}

async function bindControls() {
  const elements = Array.from(document.getElementsByClassName('control'))
  log('bind', elements.map((el) => el.id))
  for (const el of elements) {
    el.addEventListener('keyup', async (event) => {
      if (timers.id) clearTimeout(timers.id)
      timers.id = setTimeout(async () => {
        const id = event.target.id
        const data = event.target.value.trim()
        if (!ws) initWS()
        while (ws.readyState !== 1) await wait(100);
        log('ws send', id, data)
        ws.send(JSON.stringify({ id, data }))
        clearTimeout(timers.elementID)
      }, debounceTime)
    })
  }
}

async function main() {
  initWS()
  initData()
  bindControls()
  while (options?.batch === undefined) await wait(100)
  webcam = new WebCam()
  webcam.start({ element: 'webcam', canvas: 'canvas', width: options.width, height: options.height })
  drawImage() // start loop
  console.log('options', options)
  initNVML()
}

window.onload = main
