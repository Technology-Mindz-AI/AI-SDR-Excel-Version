function login() {
  let user = document.getElementById("username").value.trim();
  let pass = document.getElementById("password").value.trim();
  let error = document.getElementById("error-msg");

  let emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}$/;

  if (!emailPattern.test(user)) {
    error.textContent = "Please enter a valid email address.";
    return;
  }

  if (pass.length < 6) {
    error.textContent = "Password must be at least 6 characters.";
    return;
  }

  // Use the actual input values instead of hardcoded ones
  const email = user;
  const password = pass;

  fetch('/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    credentials: 'include', // ensures cookies are stored
    body: JSON.stringify({ email, password })
  })
  .then(response => {
    if (!response.ok) {
      return response.json().then(err => {
        // get the backend error message if available
        throw new Error(err.detail || "Login failed");
      });
    }
    return response.json();
  })
  .then(data => {
    console.log(data); // { "message": "Login successful" }
    // Instead of alert, redirect to upload page on success
    window.location.href = "/upload";
  })
  .catch(error => {
    console.error("Error:", error);
    error.textContent = "Login failed: " + error.message;
  });
}

// Add event listener to the login button
document.getElementById("loginBtn").addEventListener("click", function(event) {
  event.preventDefault(); // Prevent form submission if it's inside a form
  login();
});
