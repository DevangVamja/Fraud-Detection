const form = document.querySelector('#transaction-form');
const jsonPreview = document.querySelector('#json-preview');
const curlPreview = document.querySelector('#curl-preview');
const responseSummary = document.querySelector('#response-summary');
const latencyValue = document.querySelector('#latency-value');
const historyList = document.querySelector('#history-list');
const historyTemplate = document.querySelector('#history-item-template');
const sampleSelect = document.querySelector('#sample-presets');
const copyCurlButton = document.querySelector('#copy-curl');
const clearHistoryButton = document.querySelector('#clear-history');
const toggleThemeButton = document.querySelector('#toggle-theme');

const API_ENDPOINT = '/predict';

const presets = {
  legit: {
    step: 4,
    type: 'TRANSFER',
    amount: 850.75,
    oldbalanceOrg: 4500.2,
    newbalanceOrig: 3649.45,
    oldbalanceDest: 1200.0,
    newbalanceDest: 2050.75,
    nameOrig: 'C123456789',
    nameDest: 'M99887766',
    isFlaggedFraud: 0,
  },
  fraud: {
    step: 1,
    type: 'TRANSFER',
    amount: 9850.0,
    oldbalanceOrg: 9850.0,
    newbalanceOrig: 0.0,
    oldbalanceDest: 0.0,
    newbalanceDest: 0.0,
    nameOrig: 'C882200339',
    nameDest: 'C332200882',
    isFlaggedFraud: 1,
  },
  small: {
    step: 12,
    type: 'PAYMENT',
    amount: 49.99,
    oldbalanceOrg: 350.0,
    newbalanceOrig: 300.01,
    oldbalanceDest: 1020.0,
    newbalanceDest: 1069.99,
    nameOrig: 'C78221',
    nameDest: 'M00345',
    isFlaggedFraud: 0,
  },
};

function normaliseFormData(formData) {
  const payload = Object.fromEntries(formData.entries());
  const numericFields = [
    'step',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
  ];

  for (const field of numericFields) {
    const value = payload[field];
    payload[field] = value === '' ? null : Number(value);
  }

  payload.isFlaggedFraud = form.elements.isFlaggedFraud.checked ? 1 : 0;

  if (!payload.nameOrig) {
    payload.nameOrig = null;
  }
  if (!payload.nameDest) {
    payload.nameDest = null;
  }

  return payload;
}

function formatJson(payload) {
  return JSON.stringify(payload, null, 2);
}

function buildCurl(payload) {
  const escapedJson = formatJson(payload).replace(/"/g, '\\"');
  return [
    "curl -X POST 'http://localhost:8000" + API_ENDPOINT + "'",
    "  -H 'Content-Type: application/json'",
    "  -d \"" + escapedJson.replace(/\n/g, "\\n") + "\"",
  ].join('\n');
}

function updatePreviews() {
  const payload = normaliseFormData(new FormData(form));
  jsonPreview.textContent = formatJson(payload);
  curlPreview.textContent = buildCurl(payload);
}

function setFormValues(values) {
  for (const [key, value] of Object.entries(values)) {
    if (key === 'isFlaggedFraud') {
      form.elements[key].checked = Boolean(value);
    } else if (form.elements[key]) {
      form.elements[key].value = value;
    }
  }
  updatePreviews();
}

async function submitForm(event) {
  event.preventDefault();
  const formData = new FormData(form);
  const payload = normaliseFormData(formData);
  const startTime = performance.now();

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    const elapsed = performance.now() - startTime;
    latencyValue.textContent = `${elapsed.toFixed(0)} ms`;

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || response.statusText);
    }

    const result = await response.json();
    renderSuccess(result, payload);
  } catch (error) {
    renderError(error, payload);
  }
}

function renderSuccess(result, payload) {
  const summary = `Prediction: ${result.is_fraud ? 'Fraudulent' : 'Legitimate'} • Probability ${
    (result.fraud_probability * 100).toFixed(2)
  }%`;
  responseSummary.className = 'response-summary success';
  responseSummary.textContent = summary;
  addHistoryItem({
    payload,
    result: result.is_fraud,
    probability: result.fraud_probability,
  });
}

function renderError(error, payload) {
  responseSummary.className = 'response-summary error';
  responseSummary.textContent = `Request failed: ${error.message}`;
  addHistoryItem({ payload, result: null, probability: null, error: error.message });
}

function addHistoryItem(entry) {
  const clone = historyTemplate.content.cloneNode(true);
  const pill = clone.querySelector('[data-result]');
  const payloadLabel = clone.querySelector('.payload');
  const timeEl = clone.querySelector('[data-time]');
  const probabilityEl = clone.querySelector('.probability');

  payloadLabel.textContent = formatJson(entry.payload).replace(/\s+/g, ' ');
  timeEl.textContent = new Date().toLocaleTimeString();

  if (entry.result === null) {
    pill.style.background = 'transparent';
    pill.style.border = '1px solid var(--negative)';
    probabilityEl.textContent = 'Error';
    probabilityEl.style.color = 'var(--negative)';
  } else {
    const isFraud = Boolean(entry.result);
    pill.classList.add(isFraud ? 'negative' : 'positive');
    probabilityEl.textContent = `${(entry.probability * 100).toFixed(2)}%`;
    probabilityEl.style.color = isFraud ? 'var(--negative)' : 'var(--positive)';
  }

  historyList.prepend(clone);
}

function clearHistory() {
  historyList.innerHTML = '';
  responseSummary.className = 'response-summary';
  responseSummary.innerHTML = '<p>No requests sent yet. Configure the payload and press submit.</p>';
  latencyValue.textContent = '–';
}

function restoreCustomValues() {
  const initialValues = {
    step: 1,
    type: 'TRANSFER',
    amount: 1500,
    oldbalanceOrg: 2000,
    newbalanceOrig: 500,
    oldbalanceDest: 0,
    newbalanceDest: 1500,
    nameOrig: '',
    nameDest: '',
    isFlaggedFraud: 0,
  };
  setFormValues(initialValues);
}

function handlePresetChange(event) {
  const value = event.target.value;
  if (value === 'custom') {
    restoreCustomValues();
    return;
  }
  setFormValues(presets[value]);
}

function attachListeners() {
  form.addEventListener('input', updatePreviews);
  form.addEventListener('change', updatePreviews);
  form.addEventListener('submit', submitForm);
  form.addEventListener('reset', () => {
    setTimeout(updatePreviews, 0);
  });
  sampleSelect.addEventListener('change', handlePresetChange);
  copyCurlButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(curlPreview.textContent);
      copyCurlButton.textContent = 'Copied!';
      setTimeout(() => (copyCurlButton.textContent = 'Copy curl'), 1500);
    } catch (error) {
      copyCurlButton.textContent = 'Press Ctrl+C';
      setTimeout(() => (copyCurlButton.textContent = 'Copy curl'), 1500);
    }
  });
  clearHistoryButton.addEventListener('click', clearHistory);
  toggleThemeButton.addEventListener('click', () => {
    document.body.classList.toggle('light');
  });
}

restoreCustomValues();
attachListeners();
updatePreviews();
