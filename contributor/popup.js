/* global Blob, chrome, clearTimeout, document, setTimeout, URL, window */
const $ = (id) => document.getElementById(id);
let selected;
let prepared;
let expiryTimer;
const error = (message) => { $('error').textContent = message; $('error').hidden = !message; };
const selectionIsFresh = () => Boolean(selected && Number.isFinite(selected.expiresAt) && selected.expiresAt > Date.now());
const clearLocalReview = async (message = 'The image capture expired. Choose it again.') => {
  if (expiryTimer) clearTimeout(expiryTimer);
  expiryTimer = undefined;
  selected = undefined;
  prepared = undefined;
  $('thumb').removeAttribute('src');
  $('preview').hidden = true;
  $('image-source').textContent = '';
  $('prepare').hidden = false;
  $('export').hidden = true;
  $('selection-status').textContent = 'No image selected.';
  await chrome.storage.session.remove('contributorSelection');
  error(message);
  update();
};
const scheduleExpiry = () => {
  if (expiryTimer) clearTimeout(expiryTimer);
  const delay = Math.max(0, Math.min(selected.expiresAt - Date.now(), 2_147_483_647));
  expiryTimer = setTimeout(() => { void clearLocalReview(); }, delay);
};
const validScore = (label, raw) => {
  const score = Number(raw);
  return raw !== '' && Number.isFinite(score) && score >= 0 && score <= 100 &&
    ((label === 'ai_false_negative' && score < 65) || (label === 'real_false_positive' && score >= 65));
};
const update = () => { const label = document.querySelector('input[name=label]:checked')?.value; $('prepare').disabled = !(selected?.thumbnail && selected?.imageSha256 && validScore(label, $('score').value) && $('attest').checked && !$('evidence').validity.tooShort && !document.querySelector('.block:checked')); };
const showSelection = async (selection) => {
  try {
    if (!Number.isFinite(selection.expiresAt) || selection.expiresAt <= Date.now()) throw new Error('The image capture expired. Choose it again.');
    if (!selection.thumbnail?.startsWith('data:image/jpeg;base64,') || !/^[a-f0-9]{64}$/.test(selection.imageSha256 || '') || selection.screenshot) {
      throw new Error('The selected image was not captured safely.');
    }
    selected = selection;
    $('selection-status').textContent = 'Image captured locally.';
    $('preview').hidden = false;
    $('thumb').src = selected.thumbnail;
    $('image-source').textContent = selected.pageUrl;
    scheduleExpiry();
    update();
  } catch (captureError) {
    await chrome.storage.session.remove('contributorSelection');
    error(captureError instanceof Error ? captureError.message : 'The selected image could not be captured.');
  }
};

$('select').addEventListener('click', async () => {
  error('');
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !/^https?:$/.test(new URL(tab.url || '').protocol)) return error('Open an http or https page first.');
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    const response = await chrome.tabs.sendMessage(tab.id, { type: 'CONTRIBUTOR_START_PICKER' });
    if (!response?.started) throw new Error('No visible image is available.');
    $('selection-status').textContent = 'Choose an image on the page. Reopen this extension after selection.';
    window.close();
  } catch (pickerError) {
    error(pickerError instanceof Error ? pickerError.message : 'The image picker could not start.');
  }
});

document.querySelectorAll('input,textarea').forEach((element) => element.addEventListener('input', update));
$('prepare').addEventListener('click', async () => {
  error('');
  if (!selectionIsFresh()) return clearLocalReview();
  const label = document.querySelector('input[name=label]:checked')?.value;
  const evidence = $('evidence').value.trim();
  if (!selected?.thumbnail || !selected?.imageSha256 || !validScore(label, $('score').value) || !$('attest').checked || evidence.length < 20 || document.querySelector('.block:checked')) return error('Complete the label, matching score, attestation, and evidence. False negatives must be below 65; false positives must be 65 or higher.');
  const payload = {
    schema: 'seroslop.contributor-submission.v1', createdAt: new Date().toISOString(), label,
    score: Number($('score').value), source: selected.pageUrl, evidence,
    attestation: 'I personally know this label is correct and have the right or permission to submit it for review.',
    image: { sha256: selected.imageSha256, width: selected.width, height: selected.height, thumbnail: selected.thumbnail },
    review: { rawUpload: 'disabled', quarantine: 'not_configured', humanReview: 'required', deletion: 'required_after_review', trainingLineage: 'required' }
  };
  prepared = JSON.stringify(payload, null, 2);
  $('prepare').hidden = true;
  $('export').hidden = false;
  $('selection-status').textContent = 'Prepared locally. Nothing was uploaded.';
});
$('export').addEventListener('click', async () => {
  if (!selectionIsFresh() || !prepared) return clearLocalReview();
  const blob = new Blob([prepared], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  chrome.downloads.download({ url, filename: `seroslop-review-${Date.now()}.json`, saveAs: false }, (downloadId) => {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    if (chrome.runtime.lastError || !Number.isInteger(downloadId)) {
      error('The review file could not be downloaded. Try again.');
      return;
    }
    $('selection-status').textContent = 'Review file downloaded.';
    if (expiryTimer) clearTimeout(expiryTimer);
    expiryTimer = undefined;
    selected = undefined;
    prepared = undefined;
    $('thumb').removeAttribute('src');
    void chrome.storage.session.remove('contributorSelection');
  });
});

void chrome.storage.session.get('contributorSelection').then(({ contributorSelection }) => {
  if (contributorSelection) return showSelection(contributorSelection);
  update();
});
