/* global cancelAnimationFrame, chrome, crypto, document, location, requestAnimationFrame, setTimeout, URL, window */
(() => {
  if (globalThis.__seroslopContributorInstalled) return;
  globalThis.__seroslopContributorInstalled = true;
  let active = false;
  let candidates = [];
  let candidateSet = new Set();
  let maxCandidateCountObserved = 0;
  let index = 0;
  let outline;
  let tip;
  let controller;
  let refreshFrame = 0;
  const MAX_PICKER_CANDIDATES = 5000;
  const documentToken = [...crypto.getRandomValues(new Uint8Array(16))]
    .map((value) => value.toString(16).padStart(2, '0')).join('');

  const clear = () => {
    active = false;
    if (refreshFrame) cancelAnimationFrame(refreshFrame);
    outline?.remove();
    tip?.remove();
    controller?.remove();
    outline = tip = controller = undefined;
  };

  const pageUrl = () => {
    const value = new URL(location.href);
    value.username = '';
    value.password = '';
    value.search = '';
    value.hash = '';
    return value.origin + value.pathname;
  };

  const eligible = (image) => {
    if (!image?.isConnected || !image.complete || image.naturalWidth < 24 || image.naturalHeight < 24) return false;
    const rect = image.getBoundingClientRect();
    return rect.width >= 24 && rect.height >= 24 && rect.right > 0 && rect.bottom > 0 && rect.left < window.innerWidth && rect.top < window.innerHeight;
  };

  const collect = () => {
    const found = [];
    const limit = Math.min(document.images.length, MAX_PICKER_CANDIDATES);
    for (let item = 0; item < limit; item += 1) {
      const image = document.images.item(item);
      if (eligible(image)) found.push(image);
    }
    return found;
  };
  const imageName = (image) => (image.alt || image.getAttribute('aria-label') || 'image').trim().slice(0, 120) || 'image';

  const render = (scroll = false) => {
    const image = candidates[index];
    if (!image || !eligible(image)) return;
    if (scroll) image.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const rect = image.getBoundingClientRect();
    Object.assign(outline.style, {
      left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px`
    });
    const instruction = `${imageName(image)}, ${index + 1} of ${candidates.length}. Tab moves. Enter selects. Esc cancels.`;
    tip.textContent = instruction;
    controller.textContent = instruction;
    controller.setAttribute('aria-label', instruction);
  };

  const refresh = () => {
    refreshFrame = 0;
    const current = candidates[index];
    const next = collect();
    if (!next.length) return;
    candidates = next;
    candidateSet = new Set(candidates);
    maxCandidateCountObserved = Math.max(maxCandidateCountObserved, candidates.length);
    index = current && next.includes(current) ? next.indexOf(current) : 0;
    render();
  };

  const scheduleRefresh = () => {
    if (!active || refreshFrame) return;
    refreshFrame = requestAnimationFrame(refresh);
  };

  const choose = async (image) => {
    const rect = image.getBoundingClientRect();
    const selectedImage = {
      alt: imageName(image),
      width: image.naturalWidth || Math.round(rect.width),
      height: image.naturalHeight || Math.round(rect.height),
      pageUrl: pageUrl(),
      origin: location.origin,
      documentToken,
      rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
      viewport: { width: window.innerWidth, height: window.innerHeight }
    };
    // Do not capture the red outline, instruction bar, or keyboard controller.
    clear();
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const response = await chrome.runtime.sendMessage({
      type: 'CONTRIBUTOR_IMAGE_SELECTED',
      image: selectedImage
    }).catch(() => ({ stored: false }));
    globalThis.__seroslopContributorLastResponse = response;
    const message = document.createElement('div');
    message.textContent = response?.stored ? 'Image captured locally. Reopen the contributor extension.' : 'Image could not be captured.';
    Object.assign(message.style, { position: 'fixed', top: '16px', left: '50%', transform: 'translateX(-50%)', zIndex: '2147483647', padding: '10px 14px', borderRadius: '10px', background: '#111827', color: '#fff', font: '600 14px system-ui', boxShadow: '0 4px 18px #0005' });
    document.documentElement.append(message);
    setTimeout(() => message.remove(), 3000);
  };

  const start = () => {
    clear();
    candidates = collect();
    if (!candidates.length) return false;
    candidateSet = new Set(candidates);
    maxCandidateCountObserved = candidates.length;
    active = true;
    index = 0;
    outline = document.createElement('div');
    Object.assign(outline.style, { position: 'fixed', border: '2px solid #ef333b', outlineOffset: '2px', pointerEvents: 'none', zIndex: '2147483646' });
    tip = document.createElement('div');
    tip.setAttribute('role', 'status');
    Object.assign(tip.style, { position: 'fixed', top: '16px', left: '50%', transform: 'translateX(-50%)', zIndex: '2147483647', padding: '10px 14px', borderRadius: '10px', background: '#111827', color: '#fff', font: '600 14px system-ui', boxShadow: '0 4px 18px #0005' });
    controller = document.createElement('button');
    controller.type = 'button';
    Object.assign(controller.style, { clip: 'rect(0 0 0 0)', clipPath: 'inset(50%)', height: '1px', overflow: 'hidden', position: 'fixed', whiteSpace: 'nowrap', width: '1px' });
    document.documentElement.append(outline, tip, controller);
    render();
    controller.focus({ preventScroll: true });
    return true;
  };

  document.addEventListener('pointermove', (event) => {
    if (!active) return;
    const image = event.target.closest?.('img');
    if (!eligible(image)) return;
    if (!candidateSet.has(image)) {
      if (candidates.length < MAX_PICKER_CANDIDATES) {
        candidates.push(image);
        candidateSet.add(image);
        index = candidates.length - 1;
      } else {
        const replacementIndex = Math.min(index, candidates.length - 1);
        candidateSet.delete(candidates[replacementIndex]);
        candidates[replacementIndex] = image;
        candidateSet.add(image);
        index = replacementIndex;
      }
      maxCandidateCountObserved = Math.max(maxCandidateCountObserved, candidates.length);
    } else {
      index = candidates.indexOf(image);
    }
    render();
  }, true);
  document.addEventListener('click', (event) => {
    if (!active) return;
    const image = event.target.closest?.('img');
    if (!eligible(image)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void choose(image);
  }, true);
  document.addEventListener('keydown', (event) => {
    if (!active) return;
    if (event.key === 'Tab') {
      event.preventDefault();
      index = (index + (event.shiftKey ? candidates.length - 1 : 1)) % candidates.length;
      render(true);
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      void choose(candidates[index]);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      clear();
    }
  }, true);
  window.addEventListener('scroll', scheduleRefresh, { passive: true });
  window.addEventListener('resize', scheduleRefresh, { passive: true });
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === 'CONTRIBUTOR_START_PICKER') sendResponse({ started: start() });
    else if (message?.type === 'CONTRIBUTOR_CANCEL_PICKER') { clear(); sendResponse({ cancelled: true }); }
    else if (message?.type === 'CONTRIBUTOR_CONFIRM_DOCUMENT') {
      sendResponse({
        documentToken: message.documentToken === documentToken ? documentToken : undefined,
        origin: message.origin === location.origin ? location.origin : undefined
      });
    } else if (message?.type === 'CONTRIBUTOR_GET_DOCUMENT_STATE') {
      sendResponse({
        documentToken,
        origin: location.origin,
        pickerActive: active,
        pickerCandidateCount: candidates.length,
        pickerCandidateLimit: MAX_PICKER_CANDIDATES,
        pickerMaxCandidateCountObserved: maxCandidateCountObserved
      });
    }
  });
})();
