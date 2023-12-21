async function readCSV(filePath) {
    try {
        const response = await fetch(filePath);
        const text = await response.text();
        const lines = text.split('\n');

        const res = new Module.VecVecNum();
        for (let line of lines) {
            if (line.trim() === '') continue;

            const row = line.split(',');
            const vec = new Module.VecNum();

            for (let item of row) vec.push_back(parseFloat(item));
            res.push_back(vec);
        }

        return res;
    } catch (error) {
        console.error('Error reading CSV file:', error);
    }
}
