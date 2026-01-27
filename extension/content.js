let vidId = null, segments = [], checkTask = null;

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "INIT") {
        vidId = msg.id;
        fetch(`http://127.0.0.1:8000/video/${vidId}/segments`)
            .then(r => r.json()).then(data => {
                if (data.status === "completed") {
                    segments = data.segments;
                    drawProgressList();
                    monitor();
                    alert("🚀 ИИ подключен!");
                } else alert("⏳ Обработка...");
            });
    }
});

function drawProgressList() {
    if (document.getElementById('ai-prog')) document.getElementById('ai-prog').remove();
    const panel = document.createElement('div');
    panel.id = 'ai-prog';
    panel.style = "position:fixed;top:20px;right:20px;background:rgba(0,0,0,0.8);padding:15px;border-radius:10px;color:white;z-index:9999;font-family:sans-serif;border:1px solid #00d4ff;max-width:200px;";
    panel.innerHTML = `<h4 style="margin:0 0 10px 0;font-size:14px;color:#00d4ff">Прогресс обучения</h4><div id="plist"></div>`;
    document.body.appendChild(panel);
    updateList();
}

function updateList() {
    const list = document.getElementById('plist');
    if (!list) return;
    list.innerHTML = segments.map(s => `
        <div style="margin-bottom:5px;font-size:12px;display:flex;align-items:center;">
            <span style="color:${s.passed ? '#00ff00' : '#888'};margin-right:8px;">${s.passed ? '✔' : '○'}</span>
            <span style="text-overflow:ellipsis;overflow:hidden;white-space:nowrap;">${s.topic}</span>
        </div>
    `).join('');
}

function monitor() {
    const video = document.querySelector('video');
    if (!video) return setTimeout(monitor, 1000);
    if (checkTask) clearInterval(checkTask);
    
    checkTask = setInterval(() => {
        const t = video.currentTime;
        const unpassed = segments.find(s => t > (s.end_time + 1) && !s.passed);
        if (unpassed) {
            video.currentTime = unpassed.start_time;
            return;
        }
        const s = segments.find(s => Math.abs(t - s.end_time) < 0.5 && !s.passed);
        if (s) {
            video.pause();
            renderQuiz(s, video);
        }
    }, 500);
}

function renderQuiz(s, video) {
    if (document.getElementById('ai-ovl')) return;
    const ovl = document.createElement('div');
    ovl.id = 'ai-ovl';
    ovl.style = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);z-index:999999;display:flex;justify-content:center;align-items:center;backdrop-filter:blur(5px);";
    ovl.innerHTML = `<div style="background:#222;padding:40px;border-radius:20px;border:1px solid #00d4ff;text-align:center;color:white;font-family:sans-serif;">
        <h2 style="color:#00d4ff">${s.topic}</h2>
        <p style="font-size:18px">${s.quiz.q}</p>
        <div id="opts"></div>
    </div>`;
    document.body.appendChild(ovl);

    s.quiz.opts.forEach(o => {
        const b = document.createElement('button');
        b.innerText = o;
        b.style = "display:block;width:100%;margin:10px 0;padding:12px;background:#333;color:white;border:1px solid #444;border-radius:8px;cursor:pointer;";
        b.onclick = () => {
            fetch(`http://127.0.0.1:8000/segment/${s.id}/answer?vid=${vidId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ answer: o })
            }).then(r => r.json()).then(res => {
                if (res.correct) {
                    ovl.remove(); s.passed = true; updateList(); video.play();
                } else {
                    alert("❌ Неверно!");
                    ovl.remove(); video.currentTime = s.start_time; video.play();
                }
            });
        };
        ovl.querySelector('#opts').appendChild(b);
    });
}