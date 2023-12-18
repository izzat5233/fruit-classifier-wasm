function setupTableHeader(theadSelector, numCols, name = "") {
    const thead = d3.select(theadSelector);
    let tr = thead.selectAll("tr").join();
    if (tr.empty()) tr = thead.append('tr');

    for (let i = 0; i < numCols; i++) {
        tr.append('th').text(name + (i + 1));
    }
}

function fillNumTable(tbodySelector, vecVecNum, decimalPlaces = 4) {
    const tbody = d3.select(tbodySelector);
    for (let i = 0; i < vecVecNum.size(); ++i) {
        let tr = tbody.selectAll(`tr:nth-child(${i + 1})`);
        if (tr.empty()) tr = tbody.append('tr');

        for (let j = 0; j < vecVecNum.get(i).size(); ++j) {
            const num = vecVecNum.get(i).get(j);
            const formattedNum = typeof num === 'number' ? num.toFixed(decimalPlaces) : num;
            tr.append('td').text(formattedNum);
        }
    }
}
