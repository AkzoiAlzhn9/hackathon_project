document.getElementById('start').onclick = async () => {
    const id = document.getElementById('vid').value;
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    chrome.tabs.sendMessage(tab.id, {action: "INIT", id: id});
};