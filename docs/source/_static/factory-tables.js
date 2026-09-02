(function () {
  const tierRank = new Map([
    ["🟡", 0],
    ["🟣", 1],
    ["🔵", 2],
    ["⚪", 3],
  ]);

  function cellText(row, index) {
    const cell = row.children[index];
    return cell ? cell.textContent.trim() : "";
  }

  function sortValue(row, index, mode) {
    const text = cellText(row, index);
    if (mode === "tier") {
      return tierRank.has(text) ? tierRank.get(text) : Number.MAX_SAFE_INTEGER;
    }
    return text.toLocaleLowerCase();
  }

  function compareRows(a, b, index, mode, direction) {
    const aValue = sortValue(a, index, mode);
    const bValue = sortValue(b, index, mode);
    if (aValue < bValue) {
      return -1 * direction;
    }
    if (aValue > bValue) {
      return 1 * direction;
    }
    return 0;
  }

  function sortTable(table, index, mode, direction) {
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => compareRows(a, b, index, mode, direction));
    rows.forEach((row) => tbody.appendChild(row));
  }

  function updateHeaders(table, activeHeader, direction) {
    table.querySelectorAll("thead th").forEach((header) => {
      header.removeAttribute("aria-sort");
      header.classList.remove("factory-sort-active", "factory-sort-desc");
    });
    activeHeader.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
    activeHeader.classList.add("factory-sort-active");
    if (direction === -1) {
      activeHeader.classList.add("factory-sort-desc");
    }
  }

  function enableSorting(table) {
    const headers = Array.from(table.querySelectorAll("thead th"));
    headers.forEach((header, index) => {
      const label = header.textContent.trim().toLowerCase();
      const mode = label === "tier" ? "tier" : label === "model" || label === "name" ? "name" : null;
      if (!mode) {
        return;
      }

      header.tabIndex = 0;
      header.classList.add("factory-sortable-header");
      header.title = mode === "tier" ? "Sort by tier" : "Sort alphabetically";

      const activate = () => {
        const currentMode = table.dataset.sortIndex === String(index);
        const direction = currentMode && table.dataset.sortDirection === "asc" ? -1 : 1;
        table.dataset.sortIndex = String(index);
        table.dataset.sortDirection = direction === 1 ? "asc" : "desc";
        sortTable(table, index, mode, direction);
        updateHeaders(table, header, direction);
      };

      header.addEventListener("click", activate);
      header.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("table.sortable-factory").forEach(enableSorting);
  });
})();
