/* Crosshair reveal over the pBRDF sample renders.

   The three maps of a group are stacked in register and the view is split
   between them by one draggable handle: everything left of it shows the first
   map, the top right the second, the bottom right the third. Dragging sweeps
   the split, so any sphere can be read across all three attributes at once.
   The group toggle swaps in the polarimetric triple. */

(function () {
  'use strict';

  var ASSET_V = '?v=19';

  var GROUPS = [
    { label: 'PBR', maps: [
      { key: 'rgb', label: 'RGB' },
      { key: 'roughness', label: 'Roughness' },
      { key: 'metallicity', label: 'Metallicity' }
    ] },
    { label: 'Polarimetric', maps: [
      { key: 'dop', label: 'DoP' },
      { key: 'aolp', label: 'AoLP' },
      { key: 'cop', label: 'CoP' }
    ] }
  ];

  function init() {
    var stage = document.getElementById('pbrdfStage');
    var tabs = document.getElementById('pbrdfTabs');
    if (!stage || !tabs) return;

    var x = 50, y = 50;                      // handle position, in percent
    var group = 0;

    var layers = [0, 1, 2].map(function () {
      var img = document.createElement('img');
      img.className = 'pbrdf-layer';
      img.draggable = false;
      stage.appendChild(img);
      return img;
    });
    var tags = [0, 1, 2].map(function (i) {
      var el = document.createElement('span');
      el.className = 'pbrdf-tag tag-' + i;
      stage.appendChild(el);
      return el;
    });

    var vline = document.createElement('div');
    vline.className = 'pbrdf-split v';
    var hline = document.createElement('div');
    hline.className = 'pbrdf-split h';
    var handle = document.createElement('div');
    handle.className = 'pbrdf-handle';
    handle.setAttribute('role', 'slider');
    handle.setAttribute('aria-label', 'Move the split');
    stage.appendChild(vline);
    stage.appendChild(hline);
    stage.appendChild(handle);

    function layout() {
      // left column, then the right column halved by the horizontal line
      layers[0].style.clipPath = 'polygon(0 0, ' + x + '% 0, ' + x + '% 100%, 0 100%)';
      layers[1].style.clipPath =
        'polygon(' + x + '% 0, 100% 0, 100% ' + y + '%, ' + x + '% ' + y + '%)';
      layers[2].style.clipPath =
        'polygon(' + x + '% ' + y + '%, 100% ' + y + '%, 100% 100%, ' + x + '% 100%)';

      vline.style.left = x + '%';
      hline.style.left = x + '%';
      hline.style.top = y + '%';
      hline.style.width = (100 - x) + '%';
      handle.style.left = x + '%';
      handle.style.top = y + '%';

      // hide a label once its region is too small to hold it
      tags[0].style.display = x > 16 ? '' : 'none';
      tags[1].style.display = (100 - x) > 18 && y > 12 ? '' : 'none';
      tags[2].style.display = (100 - x) > 18 && (100 - y) > 12 ? '' : 'none';
    }

    function setGroup(g) {
      group = g;
      GROUPS[g].maps.forEach(function (m, i) {
        layers[i].src = './static/image/pbrdf/' + m.key + '.webp' + ASSET_V;
        layers[i].alt = GROUPS[g].label + ' ' + m.label;
        tags[i].textContent = m.label;
      });
      Array.prototype.forEach.call(tabs.children, function (b, i) {
        b.classList.toggle('is-active', i === g);
      });
      layout();
    }

    GROUPS.forEach(function (g, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'scene-tab';
      b.textContent = g.label;
      b.addEventListener('click', function () { setGroup(i); });
      tabs.appendChild(b);
    });

    function move(ev) {
      var rect = stage.getBoundingClientRect();
      var touch = ev.touches && ev.touches[0];
      x = ((touch ? touch.clientX : ev.clientX) - rect.left) / rect.width * 100;
      y = ((touch ? touch.clientY : ev.clientY) - rect.top) / rect.height * 100;
      x = Math.max(2, Math.min(98, x));
      y = Math.max(2, Math.min(98, y));
      layout();
    }

    var dragging = false;
    stage.addEventListener('mousedown', function (e) { e.preventDefault(); dragging = true; move(e); });
    window.addEventListener('mousemove', function (e) { if (dragging) move(e); });
    window.addEventListener('mouseup', function () { dragging = false; });
    stage.addEventListener('touchstart', function (e) { e.preventDefault(); move(e); }, { passive: false });
    stage.addEventListener('touchmove', function (e) { e.preventDefault(); move(e); }, { passive: false });

    handle.tabIndex = 0;
    handle.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 10 : 2;
      var moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0],
                    ArrowUp: [0, -step], ArrowDown: [0, step] };
      var m = moves[e.key];
      if (!m) return;
      e.preventDefault();
      x = Math.max(2, Math.min(98, x + m[0]));
      y = Math.max(2, Math.min(98, y + m[1]));
      layout();
    });

    setGroup(0);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
