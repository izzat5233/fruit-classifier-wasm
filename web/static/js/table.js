function appendTableHeaders(theadSelector, numCols, name = "") {
    const thead = d3.select(theadSelector);
    let tr = thead.selectAll("tr").join();
    if (tr.empty()) tr = thead.append('tr');

    const headers = [];
    for (let i = 0; i < numCols; i++) {
        headers.push(tr.append('th').text(name + (i + 1)));
    }
    return headers;
}

function appendTableCells(tbodySelector, vecVecNum, decimalPlaces = 4) {
    const tbody = d3.select(tbodySelector);
    const allRows = [];

    for (let i = 0; i < vecVecNum.size(); ++i) {
        let tr = tbody.selectAll(`tr:nth-child(${i + 1})`);
        if (tr.empty()) tr = tbody.append('tr');

        const rowCells = [];
        for (let j = 0; j < vecVecNum.get(i).size(); ++j) {
            const num = vecVecNum.get(i).get(j);
            const formattedNum = typeof num === 'number' ? num.toFixed(decimalPlaces) : num;
            rowCells.push(tr.append('td').text(formattedNum));
        }
        allRows.push(rowCells);
    }
    return allRows;
}

function styleTableSelection(headers, cells, className) {
    headers.forEach(header => header.classed(className, true));
    cells.forEach(row => row.forEach(cell => cell.classed(className, true)));
}