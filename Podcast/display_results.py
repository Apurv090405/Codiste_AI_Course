from flask import Flask, render_template_string, send_file
import json
import os


app = Flask(__name__)

@app.route("/")
def index():
    json_files = [f for f in os.listdir() if f.endswith(".json")]
    if not json_files:
        return "No podcast JSON files found.", 404
    
    latest_file = max(json_files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        podcast = json.load(f)
    
    html = """
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <title>Podcast Results</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Mono&family=Segoe+UI:wght@400;600&display=swap');
            :root {
                --primary: #1a73e8;
                --primary-dark: #0c47a1;
                --bg: #f6f8fa;
                --container-bg: #fff;
                --shadow: 0 4px 24px rgba(0,0,0,0.08);
                --border: #e3e8ee;
                --accent: #f3f6fb;
                --script-bg: #f9fafb;
                --script-border: #e3e8ee;
                --radius: 12px;
                --radius-sm: 8px;
                --radius-xs: 6px;
                --transition: 0.2s cubic-bezier(.4,0,.2,1);
            }
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
                background: var(--bg);
                color: #222;
                font-family: 'Segoe UI', Arial, sans-serif;
                scroll-behavior: smooth;
            }
            body {
                min-height: 100vh;
                display: flex;
                align-items: flex-start;
                justify-content: center;
            }
            .container {
                max-width: 900px;
                margin: 40px auto 40px auto;
                background: var(--container-bg);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 32px 40px 40px 40px;
                animation: fadeIn 0.7s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px);}
                to { opacity: 1; transform: none;}
            }
            h1 {
                color: var(--primary);
                margin-bottom: 10px;
                font-size: 2.2em;
                font-weight: 600;
                letter-spacing: 0.01em;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            h2 {
                color: #333;
                margin-top: 32px;
                margin-bottom: 16px;
                border-bottom: 2px solid var(--border);
                padding-bottom: 4px;
                font-size: 1.4em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            h3 {
                color: #444;
                margin-bottom: 6px;
                font-size: 1.1em;
                font-weight: 600;
            }
            .episode {
                margin-bottom: 28px;
                background: var(--accent);
                border-radius: var(--radius-sm);
                padding: 18px 22px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.03);
                transition: box-shadow var(--transition), transform var(--transition);
                border-left: 4px solid var(--primary);
            }
            .episode:hover {
                box-shadow: 0 6px 24px rgba(26,115,232,0.08);
                transform: translateY(-2px) scale(1.01);
            }
            .script {
                background: var(--script-bg);
                border: 1px solid var(--script-border);
                border-radius: var(--radius-xs);
                padding: 12px 14px;
                margin-top: 6px;
                font-family: 'Fira Mono', 'Consolas', monospace;
                font-size: 1em;
                color: #222;
                white-space: pre-wrap;
                overflow-x: auto;
                line-height: 1.6;
                transition: background 0.2s;
            }
            .script:hover {
                background: #eef2f7;
            }
            a {
                color: var(--primary);
                text-decoration: none;
                font-weight: 500;
                transition: color var(--transition), text-decoration var(--transition);
                border-bottom: 1px dashed var(--primary);
            }
            a:hover, a:focus {
                color: var(--primary-dark);
                text-decoration: underline;
                outline: none;
            }
            .meta {
                margin-bottom: 18px;
                font-size: 1.05em;
                background: #f7fafd;
                border-radius: var(--radius-xs);
                padding: 14px 18px;
                border: 1px solid var(--border);
                box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            }
            .meta p {
                margin: 6px 0;
                line-height: 1.5;
            }
            .meta strong {
                color: #555;
                margin-right: 4px;
                font-weight: 600;
            }
            pre {
                margin: 0;
            }
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                background: #e9eef3;
            }
            ::-webkit-scrollbar-thumb {
                background: #c6d4e6;
                border-radius: 6px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #b0c4de;
            }
            /* Responsive */
            @media (max-width: 700px) {
                .container {
                    padding: 16px 6vw 24px 6vw;
                }
                h1 { font-size: 1.4em; }
                h2 { font-size: 1.1em; }
                .meta { font-size: 0.98em; }
            }
            @media (max-width: 480px) {
                .container {
                    padding: 8px 2vw 16px 2vw;
                }
                h1, h2 { font-size: 1em; }
                .meta { font-size: 0.93em; }
            }
            /* Button styles */
            .download-btn {
                display: inline-block;
                background: var(--primary);
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                margin-top: 8px;
                transition: background var(--transition), box-shadow var(--transition);
                box-shadow: 0 2px 8px rgba(26,115,232,0.08);
                text-decoration: none;
            }
            .download-btn:hover, .download-btn:focus {
                background: var(--primary-dark);
                color: #fff;
                box-shadow: 0 4px 16px rgba(26,115,232,0.13);
                outline: none;
            }
            /* Icon animation */
            .fa-download {
                margin-right: 6px;
                transition: transform 0.2s;
            }
            .download-btn:hover .fa-download {
                transform: translateY(2px) scale(1.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><i class="fas fa-podcast"></i> Podcast: {{ podcast.topic }}</h1>
            <div class="meta">
                <p><strong>Description:</strong> {{ podcast.description }}</p>
                <p><strong>Duration:</strong> {{ podcast.total_duration }} minutes</p>
                <p><strong>Speakers:</strong> {{ podcast.speakers }}</p>
                <p><strong>Location:</strong> {{ podcast.location }}</p>
                <p><strong>Episodes:</strong> {{ podcast.total_episodes }}</p>
                <p>
                    <a href="/download/{{ filename }}" class="download-btn">
                        <i class="fas fa-download"></i> Download JSON
                    </a>
                </p>
            </div>
            <h2><i class="fas fa-list"></i> Structure</h2>
            {% for episode in podcast.episodes %}
            <div class="episode">
                <h3>{{ episode.title }} ({{ episode.duration|round(2) }} min, {{ episode.word_count }} words)</h3>
                {% if episode.sub_scripts %}
                    {% for sub in episode.sub_scripts %}
                    <p><strong>{{ sub.sub_episode }}</strong> ({{ sub.word_count }} words)</p>
                    <pre class="script">{{ sub.script }}</pre>
                    {% endfor %}
                {% else %}
                    <pre class="script">{{ episode.script }}</pre>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, podcast=podcast, filename=latest_file)

@app.route("/download/<filename>")
def download_file(filename):
    if not filename.endswith(".json") or not os.path.exists(filename):
        return "File not found.", 404
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)