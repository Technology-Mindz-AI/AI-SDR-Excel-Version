document.addEventListener('DOMContentLoaded', function() {
    checkAuthentication();
});

async function checkAuthentication() {
    try {
        const response = await fetch('/auth-check', {
            method: 'GET',
            credentials: 'include' 
        });
        
        if (!response.ok) {

            window.location.href = "/";
            return;
        }
        
        const data = await response.json();
        console.log('Authenticated as:', data.username);
        
    } catch (error) {
        console.error('Auth check failed:', error);
        window.location.href = "/";
    }
}

const UPLOAD_URL = "/upload-file"; 
const ADD_CALL_URL = "/add-call";
const DOWNLOAD_URL = "/download-excel";

async function uploadFile() {
    let fileInput = document.getElementById("fileUpload"); 
    let fileDetails = document.getElementById("fileDetails");

    if (!fileInput || fileInput.files.length === 0) {
        return; // No UI message needed
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch(UPLOAD_URL, {
            method: "POST",
            body: formData,
            credentials: 'include'
        });

        if (!response.ok) {
            console.error("Backend error:", await response.text());
            return;
        }

        const result = await response.json();

        if (fileDetails) {
            fileDetails.innerHTML = `<strong style="color: green;"> ${file.name} uploaded successfully!</strong>`;
            setTimeout(() => {
                fileDetails.innerHTML = '';
            }, 5000);
        }

        // Initiate call processing
        const callResponse = await fetch(ADD_CALL_URL, {
            method: "POST",
            credentials: 'include'
        });

        if (!callResponse.ok) {
            console.error("Call initiation failed:", await callResponse.text());
            return;
        }

        const callResult = await callResponse.json();
        console.log("Call processing started:", callResult);

    } catch (err) {
        console.error("Upload or call error:", err);
        // No UI error shown
    }
}

async function downloadAll() {
    try {
        const response = await fetch(DOWNLOAD_URL, {
            credentials: 'include' 
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Backend error:", errorText);
            alert(`Download failed: ${response.status} - ${errorText}`);
            return;
        }

        const contentDisposition = response.headers.get('content-disposition');
        let filename = "resultant_excel.xlsx";
        
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
            if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1].replace(/"/g, '');
            }
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        window.URL.revokeObjectURL(url);
        
    } catch (err) {
        console.error("Download error:", err);
        alert("Error downloading file! Check console for details.");
    }
}


async function logout() {
    try {
        const response = await fetch('/logout', {
            method: 'POST',
            credentials: 'include'
        });
        
        if (response.ok) {
            window.location.href = "/";
        } else {
            console.error('Logout failed');
            window.location.href = "/";
        }
    } catch (err) {
        console.error('Logout error:', err);
        window.location.href = "/";
    }
}
