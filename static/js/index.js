/* PDIR project page — marks the best and second-best entry in each table column.
   Each <th> carries data-dir="lower" or "higher" to say which way is better. */

(function () {
  'use strict';

  function highlightTable(table) {
    if (!table) return;
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

      entries[0].cell.style.fontWeight = 'bold';
      entries[0].cell.style.color = '#d93025';

      /* with only two candidates, "second best" carries no information */
      if (entries.length > 2) {
        entries[1].cell.style.fontWeight = 'bold';
        entries[1].cell.style.textDecoration = 'underline';
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    Array.prototype.slice.call(document.querySelectorAll('table.res-table'))
      .forEach(highlightTable);
  });
})();
