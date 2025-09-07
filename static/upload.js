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


const UPLOAD_URL = "/upload-file";  // Upload file to temp location
const DOWNLOAD_URL = "/download-excel";


// async function uploadFile() {
//     let fileInput = document.getElementById("fileUpload");
//     const fileDetails = document.getElementById("fileDetails") || document.createElement('div');
//     fileDetails.id = "fileDetails";

//     if (!document.body.contains(fileDetails)) {
//         document.querySelector('.file-upload-wrapper').appendChild(fileDetails);
//     }

//     if (fileInput.files.length === 0) {
//         fileDetails.innerHTML = '<span style="color: red;">Please select a file first!</span>';
//         return;
//     }

//     const formData = new FormData();
//     formData.append("file", fileInput.files[0]);

//     try {
//         fileDetails.innerHTML = '<span style="color: blue;">Uploading file...</span>';

//         const response = await fetch(UPLOAD_URL, {
//             method: "POST",
//             body: formData,
//             credentials: 'include'
//         });

//         if (!response.ok) {
//             const errorText = await response.text();
//             console.error("Upload failed:", errorText);
//             fileDetails.innerHTML = `<span style="color: red;">Upload failed: ${response.status} - ${errorText}</span>`;
//             return;
//         }

//         const result = await response.json();
//         fileDetails.innerHTML = `<strong style="color: green;">✅ File uploaded:</strong><br>${result.message}`;

//         // Enable the start button
//         document.querySelector('.callBtn').disabled = false;

//     } catch (err) {
//         console.error("Upload error:", err);
//         fileDetails.innerHTML = '<span style="color: red;">Error uploading file! Check console for details.</span>';
//     }
// }


async function downloadAll() {
}


async function logout() {
}

// async function startCallProcessing() {
// }


// document.addEventListener('DOMContentLoaded', function () {
//     checkAuthentication();

//     const fileInput = document.getElementById('fileUpload');
//     const callBtn = document.querySelector('.callBtn');

//     // Handle file selection + upload
//     fileInput.addEventListener('change', uploadFile);

//     // Handle processing click
//     callBtn.addEventListener('click', async function () {
//         const fileDetails = document.getElementById("fileDetails");
//         console.log("Start Call Processing clicked");

//         try {
//             fileDetails.innerHTML += '<br><span style="color: blue;">Processing file...</span>';

//             const response = await fetch("/add-call", {
//                 method: "POST",
//                 credentials: 'include'
//             });

//             if (!response.ok) {
//                 const errorText = await response.text();
//                 console.error("Processing failed:", errorText);
//                 fileDetails.innerHTML += `<br><span style="color: red;">Processing failed: ${response.status} - ${errorText}</span>`;
//                 return;
//             }

//             const result = await response.json();
//             fileDetails.innerHTML += `<br><strong style="color: green;">✅ Processing complete:</strong><br>${result.message}`;
//         } catch (err) {
//             console.error("Processing error:", err);
//             fileDetails.innerHTML += '<br><span style="color: red;">Error processing file! Check console for details.</span>';
//         }
//     });
// });

$(document).ready(function() {
    let $callBtn = $('.callBtn');

    $('#fileUpload').on('change', function() {
        let file = this.files[0];
        if (file) {
            let fileDiv = $(`
                <div class="uploaded-file">
                    <span>${file.name}</span>
                    <span class="remove-file">&times;</span>
                </div>
            `);

            $('#uploadedFiles').html(fileDiv);
            $callBtn.prop('disabled', false);

            fileDiv.find('.remove-file').on('click', function() {
                fileDiv.remove();
                $('#fileUpload').val('');
                $callBtn.prop('disabled', true);
            });
        }
    });
});