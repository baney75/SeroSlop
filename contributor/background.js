/* global atob, Blob, btoa, chrome, createImageBitmap, crypto, OffscreenCanvas, URL */
const hex = (bytes) => [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, '0')).join('');

async function cropScreenshot(screenshot, image) {
  const encoded = screenshot.slice(screenshot.indexOf(',') + 1);
  const binary = atob(encoded);
  const screenshotBytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) screenshotBytes[index] = binary.charCodeAt(index);
  const bitmap = await createImageBitmap(new Blob([screenshotBytes], { type: 'image/png' }));
  const scaleX = bitmap.width / image.viewport.width;
  const scaleY = bitmap.height / image.viewport.height;
  const left = Math.max(0, image.rect.left);
  const top = Math.max(0, image.rect.top);
  const right = Math.min(image.viewport.width, image.rect.left + image.rect.width);
  const bottom = Math.min(image.viewport.height, image.rect.top + image.rect.height);
  const sourceX = Math.floor(left * scaleX);
  const sourceY = Math.floor(top * scaleY);
  const sourceWidth = Math.min(bitmap.width - sourceX, Math.ceil((right - left) * scaleX));
  const sourceHeight = Math.min(bitmap.height - sourceY, Math.ceil((bottom - top) * scaleY));
  if (sourceWidth < 24 || sourceHeight < 24) throw new Error('selected image is outside the viewport');
  const outputScale = Math.min(1, 640 / Math.max(sourceWidth, sourceHeight));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(sourceWidth * outputScale)), Math.max(1, Math.round(sourceHeight * outputScale)));
  const context = canvas.getContext('2d');
  if (!context) throw new Error('crop context unavailable');
  context.drawImage(bitmap, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let thumbnailBinary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    thumbnailBinary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return {
    thumbnail: `data:image/jpeg;base64,${btoa(thumbnailBinary)}`,
    imageSha256: hex(await crypto.subtle.digest('SHA-256', bytes))
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== 'CONTRIBUTOR_IMAGE_SELECTED' || !sender.tab?.windowId) return undefined;
  void (async () => {
    let stage = 'validate';
    try {
      const token = message.image?.documentToken;
      const origin = message.image?.origin;
      const parsedOrigin = new URL(origin || 'invalid:');
      if (!/^[a-f0-9]{32}$/.test(token || '') || !['http:', 'https:'].includes(parsedOrigin.protocol) || parsedOrigin.origin !== origin) throw new Error('invalid document binding');
      const confirmDocument = async () => {
        const response = await chrome.tabs.sendMessage(sender.tab.id, {
          type: 'CONTRIBUTOR_CONFIRM_DOCUMENT', documentToken: token, origin
        });
        return response?.documentToken === token && response?.origin === origin;
      };
      const [before] = await chrome.tabs.query({ active: true, windowId: sender.tab.windowId });
      if (before?.id !== sender.tab.id || !await confirmDocument()) throw new Error('selected document changed');
      stage = 'capture';
      const screenshot = await chrome.tabs.captureVisibleTab(sender.tab.windowId, { format: 'png' });
      const [after] = await chrome.tabs.query({ active: true, windowId: sender.tab.windowId });
      if (after?.id !== sender.tab.id || !await confirmDocument()) throw new Error('selected document changed');
      stage = 'crop';
      const cropped = await cropScreenshot(screenshot, message.image);
      const image = { ...message.image };
      delete image.documentToken;
      delete image.origin;
      await chrome.storage.session.set({ contributorSelection: { ...image, ...cropped, tabId: sender.tab.id, expiresAt: Date.now() + 300_000 } });
      await chrome.alarms.create('contributor-selection-expiry', { delayInMinutes: 5 });
      sendResponse({ stored: true });
    } catch {
      await chrome.storage.session.remove('contributorSelection');
      sendResponse({ stored: false, stage });
    }
  })();
  return true;
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'contributor-selection-expiry') void chrome.storage.session.remove(['contributorSelection', 'contributorPreparedReview']);
});
