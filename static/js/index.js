/* Scene viewers and table highlighting. No dependencies. */

(function () {
  'use strict';

  /* -------- paged viewer: arrows + tab buttons swap one or more images -------- */
  function makeViewer(cfg) {
    var tabsEl = document.getElementById(cfg.tabs);
    if (!tabsEl) return;

    var index = 0;
    var buttons = cfg.items.map(function (item, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'scene-tab';
      b.textContent = item.label;
      b.addEventListener('click', function () { go(i); });
      tabsEl.appendChild(b);
      return b;
    });

    function go(i) {
      index = (i + cfg.items.length) % cfg.items.length;
      var item = cfg.items[index];
      Object.keys(cfg.targets).forEach(function (key) {
        var el = document.getElementById(cfg.targets[key]);
        if (el) el.src = item[key];
      });
      buttons.forEach(function (b, k) {
        b.classList.toggle('is-active', k === index);
      });
      if (cfg.onChange) cfg.onChange(index, item);
    }

    var prev = document.getElementById(cfg.prev);
    var next = document.getElementById(cfg.next);
    if (prev) prev.addEventListener('click', function () { go(index - 1); });
    if (next) next.addEventListener('click', function () { go(index + 1); });

    go(0);
  }

  /* ---------------- best / second-best marking ---------------- */
  function highlightTable(table) {
    var headers = Array.prototype.slice.call(table.querySelectorAll('thead th'));
    var rows = Array.prototype.slice.call(table.querySelectorAll('tbody tr'));

    headers.forEach(function (th, col) {
      var dir = th.getAttribute('data-dir');
      if (!dir) return;

      var entries = [];
      rows.forEach(function (row) {
        var cell = row.children[col];
        if (!cell) return;
        var val = parseFloat(cell.textContent);
        if (!isNaN(val)) entries.push({ cell: cell, val: val });
      });
      if (entries.length < 2) return;

      entries.sort(function (a, b) {
        return dir === 'lower' ? a.val - b.val : b.val - a.val;
      });
      entries[0].cell.classList.add('best');
      if (entries.length > 2) entries[1].cell.classList.add('second');
    });
  }

  var ASSET_V = '?v=9';

  var SCENES = [
    { label: 'Cat', key: 'scene29' },
    { label: 'Bowl', key: 'scene41' },
    { label: 'Case', key: 'scene152' },
    { label: 'Foil', key: 'scene130' },
    { label: 'Owl', key: 'scene1' }
  ];

  document.addEventListener('DOMContentLoaded', function () {
    makeViewer({
      tabs: 'resTabs', prev: 'resPrev', next: 'resNext',
      targets: { pbr: 'resPbr', relight: 'resRelight' },
      items: SCENES.map(function (s) {
        return {
          label: s.label,
          pbr: './static/image/results/' + s.key + '_pbr.webp' + ASSET_V,
          relight: './static/image/results/' + s.key + '_relight.webp' + ASSET_V
        };
      })
    });

    makeViewer({
      tabs: 'envTabs', prev: 'envPrev', next: 'envNext',
      targets: { img: 'envImg' },
      items: [1, 2, 3, 4, 5, 6].map(function (n) {
        return { label: 'Set ' + n, img: './static/image/envgrid/page' + n + '.webp' };
      })
    });

    Array.prototype.slice.call(document.querySelectorAll('table.res-table'))
      .forEach(highlightTable);
  });
})();
