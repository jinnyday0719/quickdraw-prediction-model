const drawingCanvas = document.getElementById('drawingCanvas');
const drawingCtx = drawingCanvas.getContext('2d');
const downsampledCanvas = document.getElementById('downsampledCanvas');
const downsampledCtx = downsampledCanvas.getContext('2d');
const processButton = document.getElementById('processButton');
const clearButton = document.getElementById('clearButton');
const predictionResult = document.getElementById('predictionResult');

let isDrawing = false;
let lastX = null;
let lastY = null;

let model;
let classNames = [];

function initializeCanvas() {
    drawingCtx.fillStyle = "white";
    drawingCtx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
}

initializeCanvas();

async function loadModel() {
    try {
        model = await tf.loadLayersModel('./tfjs_model/model.json');
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

async function loadClassNames() {
    try {
        const response = await fetch('./tfjs_model/class.json');
        classNames = await response.json();
    } catch (error) {
        console.error("Error loading class names:", error);
    }
}

drawingCanvas.addEventListener('mousedown', (event) => {
    isDrawing = true;
    const rect = drawingCanvas.getBoundingClientRect();
    lastX = event.clientX - rect.left;
    lastY = event.clientY - rect.top;
});

drawingCanvas.addEventListener('mouseup', () => {
    isDrawing = false;
    lastX = null;
    lastY = null;
});

drawingCanvas.addEventListener('mousemove', draw);

drawingCanvas.addEventListener('touchstart', (event) => {
    event.preventDefault();
    isDrawing = true;
    const rect = drawingCanvas.getBoundingClientRect();
    lastX = event.touches[0].clientX - rect.left;
    lastY = event.touches[0].clientY - rect.top;
});

drawingCanvas.addEventListener('touchend', () => {
    isDrawing = false;
    lastX = null;
    lastY = null;
});

drawingCanvas.addEventListener('touchmove', (event) => {
    event.preventDefault();
    if (!isDrawing) return;
    const rect = drawingCanvas.getBoundingClientRect();
    const x = event.touches[0].clientX - rect.left;
    const y = event.touches[0].clientY - rect.top;
    drawOnCanvas(x, y);
});

function draw(event) {
    if (!isDrawing) return;
    const rect = drawingCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    drawOnCanvas(x, y);
}

function drawOnCanvas(x, y) {
    drawingCtx.lineWidth = 15;
    drawingCtx.lineCap = 'round';
    drawingCtx.strokeStyle = "black";

    if (lastX !== null && lastY !== null) {
        drawingCtx.beginPath();
        drawingCtx.moveTo(lastX, lastY);
        drawingCtx.lineTo(x, y);
        drawingCtx.stroke();
    }

    lastX = x;
    lastY = y;
}

processButton.addEventListener('click', async () => {
    try {
        const p = pica();
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;

        await p.resize(drawingCanvas, tempCanvas);

        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            data[i] = 255 - data[i];
            data[i + 1] = 255 - data[i + 1];
            data[i + 2] = 255 - data[i + 2];
        }

        tempCtx.putImageData(imageData, 0, 0);

        downsampledCtx.clearRect(0, 0, 28, 28);
        downsampledCtx.drawImage(tempCanvas, 0, 0, 28, 28);

        const tensor = tf.browser.fromPixels(tempCanvas, 1)
            .toFloat()
            .div(255.0)
            .expandDims(0);

        if (model) {
            const predictions = await model.predict(tensor).data();
            const predictedIndex = predictions.indexOf(Math.max(...predictions));
            const predictedClass = classNames[predictedIndex] || `Unknown (Index: ${predictedIndex})`;
            predictionResult.textContent = `Prediction Result: ${predictedClass}`;
        } else {
            predictionResult.textContent = "Model not loaded yet!";
        }
    } catch (error) {
        console.error("Error during processing and prediction:", error);
    }
});

clearButton.addEventListener('click', () => {
    initializeCanvas();
    predictionResult.textContent = "Prediction Result: N/A";
});

loadModel();
loadClassNames();
