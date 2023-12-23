function downloadCSV(csvString, filename) {
    const blob = new Blob([csvString], {type: 'text/csv;charset=utf-8;'});
    downloadBlob(blob, filename);
}

function downloadJson(jsonObject, filename) {
    const jsonString = JSON.stringify(jsonObject, null, 2);
    const blob = new Blob([jsonString], {type: 'application/json;charset=utf-8;'});
    downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", filename);

    // Append the link to the body (required for Firefox)
    document.body.appendChild(link);

    // Simulate a click on the link and clean up after
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

async function readTextFilePath(filepath) {
    try {
        const response = await fetch(filepath);
        return await response.text();
    } catch (error) {
        console.error('Error reading file:', error);
    }
}

async function readTextFileUpload(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function (event) {
            try {
                resolve(event.target.result.toString());
            } catch (error) {
                console.error('Error reading file:', error);
                reject(error);
            }
        };

        reader.onerror = function (error) {
            console.error('Error reading file:', error);
            reject(error);
        };

        reader.readAsText(file);
    });
}

async function handleTextFileUpload(fileInput) {
    if (fileInput.files.length === 0) {
        console.log('No file selected!');
        return;
    }
    try {
        const file = fileInput.files[0];
        return await readTextFileUpload(file).then(res => res.toString());
    } catch (error) {
        console.error('Error reading the file:', error);
    }
}