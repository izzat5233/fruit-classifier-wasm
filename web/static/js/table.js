function appendTableHeaders(theadSelector, values) {
    const thead = d3.select(theadSelector);
    let tr = thead.selectAll("tr").join();
    if (tr.empty()) tr = thead.append('tr');

    const headers = [];
    for (let name of values) {
        headers.push(tr.append('th').text(name));
    }
    return headers;
}

function appendTableCells(tbodySelector, arrArr, decimalPlaces = 4) {
    const tbody = d3.select(tbodySelector);
    const allRows = [];

    for (let i = 0; i < arrArr.length; i++) {
        let tr = tbody.selectAll(`tr:nth-child(${i + 1})`);
        if (tr.empty()) tr = tbody.append('tr');

        const rowCells = [];
        for (let item of arrArr[i]) {
            const formatted = typeof item === 'number' ? item.toFixed(decimalPlaces) : item;
            rowCells.push(tr.append('td').text(formatted));
        }
        allRows.push(rowCells);
    }
    return allRows;
}

function styleTableSelection(headers, cells, className) {
    headers.forEach(header => header.classed(className, true));
    cells.forEach(row => row.forEach(cell => cell.classed(className, true)));
}