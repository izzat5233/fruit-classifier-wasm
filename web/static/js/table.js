function fillNumTable(tbodySelector, vecVecNum, decimalPlaces = 4) {
    const tbody = d3.select(tbodySelector);
    for (let i = 0; i < vecVecNum.size(); i++) {
        const tr = tbody.append('tr');
        for (let j = 0; j < vecVecNum.get(i).size(); j++) {
            const num = vecVecNum.get(i).get(j);
            const formattedNum = typeof num === 'number' ? num.toFixed(decimalPlaces) : num;
            tr.append('td').text(formattedNum);
        }
    }
}

function setupTableHeader(theadSelector, numCols, name = "") {
    const thead = d3.select(theadSelector);
    const tr = thead.append('tr');
    for (let i = 0; i < numCols; i++) {
        tr.append('th').text(name + (i + 1)); // Headers numbered 1, 2, 3, ...
    }
}
