<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $prompt = trim($_POST["prompt"]);

    // FastAPI server URL
    $api_url = "http://127.0.0.1:8000/generate"; // Make sure this matches your FastAPI server

    $data = json_encode(["prompt" => $prompt]);

    $ch = curl_init($api_url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        "Content-Type: application/json"
    ]);

    $response = curl_exec($ch);
    $http_status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    // If FastAPI returns an error, handle it
    if ($http_status !== 200) {
        echo json_encode(["response" => "Error: Unable to generate response."]);
    } else {
        echo $response; // Return API's response
    }
}
?>
