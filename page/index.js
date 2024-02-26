import { WebCam } from "./webcam.js";
import { initNVML } from "./nvml.js";

const clientID = Date.now();
const debounceTime = 250;
const timers = {};
let ws = null;
let data = null;
let options = {};
let webcam = null;
let image = null;
let images = [];
let tmpCanvas = null;
let tmpCtx = null;
const receiveTimes = [];

const wait = (ms) => new Promise((res) => setTimeout(res, ms));

window.log = (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (msg) console.log(ts, ...msg); // eslint-disable-line no-console
}

const snapImage = (el) => {
  if (!el) return undefined;
  if (!tmpCanvas) tmpCanvas = document.createElement('canvas');
  if (el.videoWidth !== tmpCanvas.width || el.videoHeight !== tmpCanvas.height) {
    tmpCanvas.width = el.videoWidth;
    tmpCanvas.height = el.videoHeight; 
  };
  if (!tmpCtx) tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true})
  tmpCtx.drawImage(el, 0, 0, el.width, el.height);
  return tmpCanvas.toDataURL('image/jpeg', 0.8);
}

async function sendImage() {
  if (data.readyState !== 1) await wait(250);
  if (!data) initData()
  const input = document.getElementById('input')
  if (input.paused) return;
  const jpeg = snapImage(input)
  if (!jpeg) return;
  const encoded = new TextEncoder().encode(jpeg);
  if (data.readyState === 1) data.send(encoded)
}

async function initWS() {
  ws = new WebSocket(`ws://localhost:8000/ws/${clientID}`);
  log('ws init', ws)
  ws.onmessage = (event) => {
    const json = JSON.parse(event.data)
    if (json.ready) sendImage() // send new image on ready
    else {
      for (const [id, data] of Object.entries(json)) {
        const el = document.getElementById(id)
        if (el) el.value = data // update bound dom elements
        options[id] = data // update options
      }
    }
  };
  setInterval(() => ws.readyState === 1 ? ws.send(JSON.stringify({ ready: true })) : {}, 100); // poll for ready
}

async function initData() {
  data = new WebSocket(`ws://localhost:8000/data/${clientID}`);
  log('ws data', data)
  let previousReceiveTime = Date.now();
  data.onmessage = (event) => {
    // log('data receive', event.data)
    receiveTimes.push(Date.now() - previousReceiveTime);
    previousReceiveTime = Date.now();
    if (receiveTimes.length > 2 * options.batch * options.buffers) receiveTimes.shift();
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
  const average = receiveTimes.length > 0 ? receiveTimes.reduce((acc, cur) => acc + cur, 0) / receiveTimes.length : 100;
  setTimeout(drawImage, average);
  if (images.length === 0) return;
  image = images.shift();
  const canvas = document.getElementById('output');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  document.getElementById('fps').innerText = `FPS: ${(1000 / average).toFixed(1).padStart(4, '0')}`;
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

async function uploadVideo(evt) {
  const input = document.getElementById('input')
  const file = evt.target.files[0]; // 1st member in files-collection
  const fileURL = window.URL.createObjectURL(file);
  input.src=fileURL;
  log('video', fileURL)
  input.onloadeddata = () => {
    log('video loaded', input.videoWidth, input.videoHeight)
    document.getElementById('width').value = input.videoWidth
    document.getElementById('height').value = input.videoHeight
    data = { width: input.videoWidth, height: input.videoHeight }
    ws.send(JSON.stringify({ id: 'width', data: input.videoWidth }))
    ws.send(JSON.stringify({ id: 'height', data: input.videoHeight }))
  }
  input.load();
  input.play();
}

async function bindToolbar() {
  const input = document.getElementById('input')
  const el = {
    video: document.getElementById('select-video'),
    webcam: document.getElementById('btn-webcam'),
    pause: document.getElementById('btn-pause'),
  }
  log('toolbar', el)
  el.webcam.onclick = () => webcam.start({ element: 'input', canvas: 'canvas', width: options.width, height: options.height, 'crop': true });
  el.pause.onclick = () => input.paused ? input.play() : input.pause();
  el.video.onchange = uploadVideo;
}

async function main() {
  initWS()
  initData()
  bindControls()
  bindToolbar()
  while (options?.batch === undefined) await wait(100)
  webcam = new WebCam()
  drawImage() // start loop
  log('options', options)
  initNVML()
}

window.onload = main
