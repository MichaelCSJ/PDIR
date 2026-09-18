/* Two-axis pad for browsing the pBRDF sample renders.

   Columns are the three channels of a group, rows are the two groups: PBR
   appearance on top, polarimetric below. Drag the handle, click a cell, or use
   the arrow keys; the handle snaps to the nearest cell and the large view
   follows. All six renders are registered to one frame, so nothing shifts as
   you move between them. */

(function () {
  'use strict';

  var ASSET_V = '?v=18';

  var GROUPS = [
    { label: 'PBR', cells: [
      { key: 'rgb', label: 'RGB', note: 'Appearance under unpolarized light' },
      { key: 'roughness', label: 'Roughness', note: 'Fitted roughness' },
      { key: 'metallicity', label: 'Metallicity', note: 'Fitted metallicity' }
    ] },
    { label: 'Polarimetric', cells: [
      { key: 'dop', label: 'DoP', note: 'Degree of polarization' },
      { key: 'aolp', label: 'AoLP', note: 'Angle of linear polarization' },
      { key: 'cop', label: 'CoP', note: 'Chirality of polarization' }
    ] }
  ];

  var COLS = GROUPS[0].cells.length;
  var ROWS = GROUPS.length;

  function init() {
    var pad = document.getElementById('pbrdfPad');
    var img = document.getElementById('pbrdfImg');
    var caption = document.getElementById('pbrdfCaption');
    if (!pad || !img) return;

    var col = 0, row = 0;
    var handle = document.createElement('div');
    handle.className = 'pad-handle';

    /* one labelled cell per map, laid out by the stylesheet's grid */
    var cells = [];
    GROUPS.forEach(function (group, r) {
      group.cells.forEach(function (cell, c) {
        var el = document.createElement('button');
        el.type = 'button';
        el.className = 'pad-cell';
        el.textContent = cell.label;
        el.setAttribute('aria-label', group.label + ' ' + cell.label);
        el.addEventListener('click', function () { select(c, r); });
        pad.appendChild(el);
        cells.push({ el: el, c: c, r: r });
      });
    });
    pad.appendChild(handle);

    function select(c, r) {
      col = Math.max(0, Math.min(COLS - 1, c));
      row = Math.max(0, Math.min(ROWS - 1, r));
      var cell = GROUPS[row].cells[col];

      img.src = './static/image/pbrdf/' + cell.key + '.webp' + ASSET_V;
      img.alt = GROUPS[row].label + ' ' + cell.label;
      if (caption) {
        caption.textContent = GROUPS[row].label + ' · ' + cell.label + ' — ' + cell.note;
      }

      cells.forEach(function (x) {
        x.el.classList.toggle('is-active', x.c === col && x.r === row);
      });
      handle.style.left = ((col + 0.5) / COLS * 100) + '%';
      handle.style.top = ((row + 0.5) / ROWS * 100) + '%';
    }

    /* dragging snaps to whichever cell the pointer is over */
    function pick(ev) {
      var rect = pad.getBoundingClientRect();
      var touch = ev.touches && ev.touches[0];
      var x = ((touch ? touch.clientX : ev.clientX) - rect.left) / rect.width;
      var y = ((touch ? touch.clientY : ev.clientY) - rect.top) / rect.height;
      select(Math.floor(x * COLS), Math.floor(y * ROWS));
    }

    var dragging = false;
    pad.addEventListener('mousedown', function (e) { dragging = true; pick(e); });
    window.addEventListener('mousemove', function (e) { if (dragging) pick(e); });
    window.addEventListener('mouseup', function () { dragging = false; });
    pad.addEventListener('touchstart', function (e) { e.preventDefault(); pick(e); }, { passive: false });
    pad.addEventListener('touchmove', function (e) { e.preventDefault(); pick(e); }, { passive: false });

    pad.tabIndex = 0;
    pad.addEventListener('keydown', function (e) {
      var moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      var m = moves[e.key];
      if (!m) return;
      e.preventDefault();
      select(col + m[0], row + m[1]);
    });

    select(0, 0);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
