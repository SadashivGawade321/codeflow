/**
 * CodeFlow AI - Code Compiler with Gemini AI Chat
 */

// ============================================================================
// Configuration & State
// ============================================================================

const API_BASE_URL = '/api';

const state = {
    code: '',
    language: 'python',
    inputValues: null,
    steps: [],
    explanations: [],
    complexity: null,
    currentStepIndex: 0,
    isPlaying: false,
    playbackSpeed: 1,
    playbackInterval: null,
    aiAvailable: true,
    exampleFilter: 'all',
    examples: []
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    codeEditor: document.getElementById('code-editor'),
    lineNumbers: document.getElementById('line-numbers'),
    inputValues: document.getElementById('input-values'),
    analyzeBtn: document.getElementById('analyze-btn'),
    runBtn: document.getElementById('run-btn'),
    clearBtn: document.getElementById('clear-btn'),
    clearOutputBtn: document.getElementById('clear-output-btn'),
    
    playPauseBtn: document.getElementById('play-pause'),
    stepBackBtn: document.getElementById('step-back'),
    stepForwardBtn: document.getElementById('step-forward'),
    speedSlider: document.getElementById('speed-slider'),
    speedValue: document.getElementById('speed-value'),
    
    progressFill: document.getElementById('progress-fill'),
    currentStep: document.getElementById('current-step'),
    totalSteps: document.getElementById('total-steps'),
    
    currentCode: document.getElementById('current-code'),
    variablesGrid: document.getElementById('variables-grid'),
    arraySection: document.getElementById('array-section'),
    arrayContainer: document.getElementById('array-container'),
    arrayIndices: document.getElementById('array-indices'),
    flowchart: document.getElementById('flowchart'),
    timeline: document.getElementById('timeline'),
    outputSection: document.getElementById('output-section'),
    outputContent: document.getElementById('output-content'),
    
    stepBadge: document.getElementById('step-badge'),
    stepType: document.getElementById('step-type'),
    explanationSummary: document.getElementById('explanation-summary'),
    explanationDetailed: document.getElementById('explanation-detailed'),
    explanationWhy: document.getElementById('explanation-why'),
    explanationNotice: document.getElementById('explanation-notice'),
    explanationMistakes: document.getElementById('explanation-mistakes'),
    complexityText: document.getElementById('complexity-text'),
    
    askInput: document.getElementById('ask-input'),
    askBtn: document.getElementById('ask-btn'),
    chatMessages: document.getElementById('chat-messages'),
    
    timeComplexity: document.getElementById('time-complexity'),
    spaceComplexity: document.getElementById('space-complexity'),
    complexityExplanation: document.getElementById('complexity-explanation'),
    
    examplesLink: document.getElementById('examples-link'),
    examplesModal: document.getElementById('examples-modal'),
    closeModal: document.getElementById('close-modal'),
    examplesGrid: document.getElementById('examples-grid'),
    
    helpLink: document.getElementById('help-link'),
    helpModal: document.getElementById('help-modal'),
    closeHelpModal: document.getElementById('close-help-modal'),
    
    loadingOverlay: document.getElementById('loading-overlay'),
    toastContainer: document.getElementById('toast-container'),
    styleSelect: document.getElementById('style-select'),
    languageSelect: document.getElementById('language-select'),
    aiStatus: document.getElementById('ai-status'),
    
    // AI Panel Toggle
    aiPanel: document.getElementById('ai-panel'),
    chatToggleBtn: document.getElementById('chat-toggle-btn'),
    aiCloseBtn: document.getElementById('ai-close-btn'),
    
    // Theme Toggle
    themeToggle: document.getElementById('theme-toggle')
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEditor();
    initializeEventListeners();
    initializeTheme();
    loadExamples();
    updateLineNumbers();
    
    // Gemini AI is always available with direct API key
    if (elements.aiStatus) {
        elements.aiStatus.classList.add('active');
        elements.aiStatus.title = 'Gemini AI Connected';
    }
});

// ============================================================================
// Theme Toggle
// ============================================================================

function initializeTheme() {
    // Load saved theme or default to dark
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
    
    // Add click handler
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', toggleTheme);
    }
}

function setTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    localStorage.setItem('theme', theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

function initializeEditor() {
    if (!elements.codeEditor.value.trim()) {
        setDefaultCode('python');
    }
    updateLineNumbers();
}

function setDefaultCode(language) {
    const defaults = {
        python: {
            code: `# Python Code - Click "Run Code" to execute
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", arr)
result = bubble_sort(arr.copy())
print("Sorted array:", result)`,
            input: ''
        },
        javascript: {
            code: `// JavaScript Code - Click "Run Code" to execute
function bubbleSort(arr) {
    const n = arr.length;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

const arr = [64, 34, 25, 12, 22, 11, 90];
console.log("Original array:", arr);
const result = bubbleSort([...arr]);
console.log("Sorted array:", result);`,
            input: ''
        },
        java: {
            code: `// Java Code
import java.util.Arrays;

public class Main {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    public static void main(String[] args) {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        System.out.println("Original: " + Arrays.toString(arr));
        bubbleSort(arr);
        System.out.println("Sorted: " + Arrays.toString(arr));
    }
}`,
            input: ''
        },
        cpp: {
            code: `// C++ Code
#include <iostream>
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    cout << "Original: ";
    for(int x : arr) cout << x << " ";
    cout << endl;
    
    bubbleSort(arr);
    
    cout << "Sorted: ";
    for(int x : arr) cout << x << " ";
    cout << endl;
    return 0;
}`,
            input: ''
        }
    };
    
    const defaultLang = defaults[language] || defaults.python;
    elements.codeEditor.value = defaultLang.code;
    elements.inputValues.value = defaultLang.input;
    state.language = language;
    updateLineNumbers();
}

function initializeEventListeners() {
    elements.codeEditor.addEventListener('input', updateLineNumbers);
    elements.codeEditor.addEventListener('scroll', syncScroll);
    elements.codeEditor.addEventListener('keydown', handleTab);
    
    elements.analyzeBtn.addEventListener('click', analyzeCode);
    elements.clearBtn.addEventListener('click', clearEditor);
    
    // Run Code button for compiler
    if (elements.runBtn) {
        elements.runBtn.addEventListener('click', runCode);
    }
    
    // Clear output button
    if (elements.clearOutputBtn) {
        elements.clearOutputBtn.addEventListener('click', clearOutput);
    }
    
    elements.playPauseBtn.addEventListener('click', togglePlayback);
    elements.stepBackBtn.addEventListener('click', stepBackward);
    elements.stepForwardBtn.addEventListener('click', stepForward);
    elements.speedSlider.addEventListener('input', updateSpeed);
    
    // Gemini AI chat
    elements.askBtn.addEventListener('click', askGemini);
    elements.askInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askGemini();
    });
    
    // AI Panel Toggle (floating chat button)
    if (elements.chatToggleBtn && elements.aiPanel) {
        elements.chatToggleBtn.addEventListener('click', () => {
            elements.aiPanel.classList.toggle('collapsed');
            elements.chatToggleBtn.classList.toggle('active');
        });
    }
    
    // AI Panel Close Button
    if (elements.aiCloseBtn && elements.aiPanel) {
        elements.aiCloseBtn.addEventListener('click', () => {
            elements.aiPanel.classList.add('collapsed');
            if (elements.chatToggleBtn) {
                elements.chatToggleBtn.classList.remove('active');
            }
        });
    }
    
    // Language selector
    if (elements.languageSelect) {
        elements.languageSelect.addEventListener('change', (e) => {
            const newLang = e.target.value;
            if (confirm(`Switch to ${newLang.toUpperCase()}? This will load sample ${newLang} code.`)) {
                setDefaultCode(newLang);
            } else {
                e.target.value = state.language;
            }
        });
    }
    
    // Examples modal
    elements.examplesLink.addEventListener('click', (e) => {
        e.preventDefault();
        elements.examplesModal.classList.add('show');
    });
    elements.closeModal.addEventListener('click', () => {
        elements.examplesModal.classList.remove('show');
    });
    
    // Help modal
    if (elements.helpLink) {
        elements.helpLink.addEventListener('click', (e) => {
            e.preventDefault();
            elements.helpModal.classList.add('show');
        });
    }
    if (elements.closeHelpModal) {
        elements.closeHelpModal.addEventListener('click', () => {
            elements.helpModal.classList.remove('show');
        });
    }
    
    // Modal overlay close
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', () => {
            elements.examplesModal?.classList.remove('show');
            elements.helpModal?.classList.remove('show');
        });
    });
    
    // Language filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.exampleFilter = e.target.dataset.lang;
            filterExamples();
        });
    });
    
    if (elements.styleSelect) {
        elements.styleSelect.addEventListener('change', updateStyle);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

function handleKeyboardShortcuts(e) {
    // Ctrl+Enter to run code
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        runCode();
    }
    // Ctrl+Shift+Enter to analyze
    if (e.ctrlKey && e.shiftKey && e.key === 'Enter') {
        e.preventDefault();
        analyzeCode();
    }
    // Ctrl+L to clear
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        clearEditor();
    }
    // Arrow keys for stepping (when not in input)
    if (document.activeElement !== elements.codeEditor && 
        document.activeElement !== elements.askInput &&
        document.activeElement !== elements.inputValues) {
        if (e.key === 'ArrowRight') stepForward();
        if (e.key === 'ArrowLeft') stepBackward();
        if (e.key === ' ' && e.target.tagName !== 'BUTTON') {
            e.preventDefault();
            togglePlayback();
        }
    }
}

// ============================================================================
// Code Compiler - Run Code
// ============================================================================

async function runCode() {
    const code = elements.codeEditor.value.trim();
    if (!code) {
        showToast('Please enter some code', 'error');
        return;
    }
    
    const language = elements.languageSelect?.value || 'python';
    showLoading(true);
    
    try {
        if (language === 'javascript') {
            // Run JavaScript directly in browser
            await runJavaScript(code);
        } else {
            // Use Gemini AI to simulate execution for other languages
            await runWithGemini(code, language);
        }
        
        // Award XP for running code
        gamification.awardXP('run_code', language);
    } catch (error) {
        displayOutput(`Error: ${error.message}`, 'error');
    }
    
    showLoading(false);
}

async function runJavaScript(code) {
    const logs = [];
    const originalLog = console.log;
    const originalError = console.error;
    const originalWarn = console.warn;
    
    console.log = (...args) => {
        logs.push({ type: 'log', content: args.map(formatLogValue).join(' ') });
    };
    console.error = (...args) => {
        logs.push({ type: 'error', content: args.map(formatLogValue).join(' ') });
    };
    console.warn = (...args) => {
        logs.push({ type: 'warn', content: args.map(formatLogValue).join(' ') });
    };
    
    try {
        const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
        const fn = new AsyncFunction(code);
        const result = await fn();
        
        if (result !== undefined) {
            logs.push({ type: 'result', content: `Return: ${formatLogValue(result)}` });
        }
        
        if (logs.length === 0) {
            logs.push({ type: 'info', content: '✓ Code executed successfully (no output)' });
        }
        
        displayOutput(logs);
        showToast('JavaScript executed', 'success');
    } catch (error) {
        logs.push({ type: 'error', content: `${error.name}: ${error.message}` });
        displayOutput(logs);
        showToast('Execution error', 'error');
    } finally {
        console.log = originalLog;
        console.error = originalError;
        console.warn = originalWarn;
    }
}

function formatLogValue(val) {
    if (val === null) return 'null';
    if (val === undefined) return 'undefined';
    if (typeof val === 'object') {
        try {
            return JSON.stringify(val, null, 2);
        } catch {
            return String(val);
        }
    }
    return String(val);
}

async function runWithGemini(code, language) {
    try {
        const response = await fetch(`${API_BASE_URL}/run-code`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, language })
        });
        
        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        
        if (data.success) {
            displayOutput([{ type: 'log', content: data.output }]);
            showToast(`${language.toUpperCase()} executed`, 'success');
        } else {
            displayOutput([{ type: 'error', content: data.output || data.error }]);
            showToast('Execution failed', 'error');
        }
    } catch (error) {
        displayOutput([{ type: 'error', content: `Failed to run code: ${error.message}` }]);
        showToast('Execution failed', 'error');
    }
}

function displayOutput(logs, type = null) {
    if (!elements.outputContent) return;
    
    if (typeof logs === 'string') {
        logs = [{ type: type || 'log', content: logs }];
    }
    
    const html = logs.map(log => {
        const className = `output-line output-${log.type}`;
        const icon = getOutputIcon(log.type);
        return `<div class="${className}">${icon}<pre>${escapeHtml(log.content)}</pre></div>`;
    }).join('');
    
    elements.outputContent.innerHTML = html || '<span class="output-placeholder">No output</span>';
}

function getOutputIcon(type) {
    const icons = {
        log: '<span class="output-icon">→</span>',
        error: '<span class="output-icon error">✗</span>',
        warn: '<span class="output-icon warn">⚠</span>',
        result: '<span class="output-icon result">←</span>',
        info: '<span class="output-icon info">ℹ</span>'
    };
    return icons[type] || icons.log;
}

function clearOutput() {
    if (elements.outputContent) {
        elements.outputContent.innerHTML = '<span class="output-placeholder">Run your code to see output here...</span>';
    }
    showToast('Output cleared', 'info');
}

// ============================================================================
// Gemini AI Chat
// ============================================================================

async function askGemini() {
    const question = elements.askInput.value.trim();
    if (!question) return;
    
    addChatMessage(question, 'user');
    elements.askInput.value = '';
    
    const typingId = showTypingIndicator();
    
    const code = elements.codeEditor.value;
    const language = elements.languageSelect?.value || 'python';

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                code: code,
                language: language
            })
        });
        
        removeTypingIndicator(typingId);
        
        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        const answer = data.response || data.answer || 'Sorry, I could not generate a response.';
        
        addChatMessage(answer, 'ai');
        
        // Award XP for asking AI
        gamification.awardXP('ask_ai', language);
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
        console.error('AI Chat error:', error);
    }
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message ai-message typing';
    div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    elements.chatMessages.appendChild(div);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ============================================================================
// Editor Functions
// ============================================================================

function updateLineNumbers() {
    const lines = elements.codeEditor.value.split('\n');
    elements.lineNumbers.innerHTML = lines
        .map((_, i) => `<div>${i + 1}</div>`)
        .join('');
}

function syncScroll() {
    elements.lineNumbers.scrollTop = elements.codeEditor.scrollTop;
}

function handleTab(e) {
    if (e.key === 'Tab') {
        e.preventDefault();
        const start = elements.codeEditor.selectionStart;
        const end = elements.codeEditor.selectionEnd;
        elements.codeEditor.value = 
            elements.codeEditor.value.substring(0, start) +
            '    ' +
            elements.codeEditor.value.substring(end);
        elements.codeEditor.selectionStart = elements.codeEditor.selectionEnd = start + 4;
        updateLineNumbers();
    }
}

function clearEditor() {
    elements.codeEditor.value = '';
    elements.inputValues.value = '';
    updateLineNumbers();
    resetVisualization();
    if (elements.outputContent) {
        elements.outputContent.innerHTML = '<span class="output-placeholder">Run your code to see output here...</span>';
    }
    showToast('Editor cleared', 'info');
}

function resetVisualization() {
    state.steps = [];
    state.explanations = [];
    state.currentStepIndex = 0;
    state.isPlaying = false;
    
    if (state.playbackInterval) {
        clearInterval(state.playbackInterval);
    }
    
    elements.currentCode.innerHTML = '<code>Click "Run Code" to execute or "Analyze" to visualize</code>';
    elements.variablesGrid.innerHTML = '<div class="empty-state">No variables</div>';
    elements.arraySection.style.display = 'none';
    elements.flowchart.innerHTML = '';
    elements.timeline.innerHTML = '';
    elements.progressFill.style.width = '0%';
    elements.currentStep.textContent = '0';
    elements.totalSteps.textContent = '0';
    
    updatePlayIcon(false);
}

// ============================================================================
// API Functions
// ============================================================================

async function analyzeCode() {
    const code = elements.codeEditor.value.trim();
    if (!code) {
        showToast('Please enter some code', 'error');
        return;
    }
    
    showLoading(true);
    
    let inputValues = null;
    try {
        const inputText = elements.inputValues.value.trim();
        if (inputText) {
            inputValues = JSON.parse(inputText);
        }
    } catch (e) {
        console.warn('Invalid input values JSON');
    }
    
    // Get selected language
    const language = elements.languageSelect?.value || 'python';
    state.language = language;
    
    try {
        // For Python, use the full step-by-step analysis
        if (language === 'python') {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, language, input_values: inputValues })
            });
            
            if (!response.ok) throw new Error('Analysis failed');
            
            const data = await response.json();
            
            if (!data.success) {
                showToast(data.error || 'Analysis failed', 'error');
                showLoading(false);
                return;
            }
            
            state.steps = data.steps;
            state.explanations = data.explanations;
            state.complexity = data.complexity;
            state.currentStepIndex = 0;
            
            renderVisualization();
            updateComplexityDisplay();
            
            showToast('Analysis complete', 'success');
        } else {
            // For other languages, use AI explanation
            const response = await fetch(`${API_BASE_URL}/explain-code`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, language })
            });
            
            if (!response.ok) throw new Error('Analysis failed');
            
            const data = await response.json();
            
            if (!data.success) {
                // Show error but try to display what we can
                if (data.error) {
                    showToast('AI not configured. Showing basic analysis.', 'info');
                }
            }
            
            // Display AI explanation in chat
            displayAIExplanation(data);
            
            showToast(`${language.toUpperCase()} code analyzed with AI`, 'success');
        }
        
        // Award XP for analyzing code
        gamification.awardXP('analyze_code', language);
    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Analysis failed: ' + error.message, 'error');
    }
    
    showLoading(false);
}

function displayAIExplanation(data) {
    // Clear previous state
    resetVisualization();
    
    // Update the current code display
    elements.currentCode.innerHTML = `<code>${escapeHtml(elements.codeEditor.value.substring(0, 500))}...</code>`;
    
    // Update explanation panel with AI data
    elements.stepBadge.textContent = 'AI Analysis';
    elements.stepType.textContent = data.algorithm_type || 'Code Analysis';
    elements.explanationSummary.textContent = data.summary || 'Analysis complete';
    elements.explanationDetailed.textContent = data.detailed_explanation || 'No detailed explanation available';
    elements.explanationWhy.textContent = 'See explanation above for details';
    
    // Key concepts
    if (data.key_concepts && data.key_concepts.length > 0) {
        elements.explanationNotice.innerHTML = data.key_concepts
            .map(c => `<li>${c}</li>`).join('');
    } else {
        elements.explanationNotice.innerHTML = '<li>Ask AI for more details</li>';
    }
    
    // Common mistakes
    if (data.common_mistakes && data.common_mistakes.length > 0) {
        elements.explanationMistakes.innerHTML = data.common_mistakes
            .map(m => `<li>${m}</li>`).join('');
    }
    
    // Complexity
    if (data.time_complexity) {
        elements.timeComplexity.textContent = data.time_complexity;
        elements.spaceComplexity.textContent = data.space_complexity || 'Unknown';
        elements.complexityText.textContent = data.time_complexity;
    }
    
    // Add to chat
    const summary = data.summary || 'Code analysis complete';
    addChatMessage(`Analysis for your ${state.language} code:\n\n${summary}`, 'ai');
    
    // Optimization tips if available
    if (data.optimization_tips && data.optimization_tips.length > 0) {
        addChatMessage(`💡 Optimization Tips:\n• ${data.optimization_tips.join('\n• ')}`, 'ai');
    }
}

function addChatMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Format markdown-like content
    let formattedText = escapeHtml(text);
    
    // Convert code blocks
    formattedText = formattedText.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
    });
    
    // Convert inline code
    formattedText = formattedText.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert bold
    formattedText = formattedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert bullet points
    formattedText = formattedText.replace(/^• (.+)$/gm, '<li>$1</li>');
    formattedText = formattedText.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    
    // Convert newlines
    formattedText = formattedText.replace(/\n/g, '<br>');
    
    messageDiv.innerHTML = `<div class="message-content">${formattedText}</div>`;
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

async function updateStyle() {
    const style = elements.styleSelect.value;
    try {
        await fetch(`${API_BASE_URL}/settings?style=${style}`, { method: 'PUT' });
        showToast(`Difficulty set to ${style}`, 'success');
    } catch (e) {
        console.error('Failed to update style');
    }
}

async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE_URL}/examples`);
        const data = await response.json();
        
        // Store examples in state for filtering
        state.examples = data.examples || [];
        renderExamples(state.examples);
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

function renderExamples(examples) {
    elements.examplesGrid.innerHTML = examples.map(ex => {
        const lang = ex.language || 'python';
        const langDisplay = {
            'python': 'Python',
            'javascript': 'JS',
            'java': 'Java',
            'cpp': 'C++'
        }[lang] || lang.toUpperCase();
        
        return `
            <div class="example-card" 
                data-code="${encodeURIComponent(ex.code)}" 
                data-language="${lang}"
                data-input="${encodeURIComponent(JSON.stringify(ex.input_values || {}))}">
                <span class="language-badge">${langDisplay}</span>
                <h3>${ex.name}</h3>
                <p>${ex.description}</p>
            </div>
        `;
    }).join('');
    
    document.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            const code = decodeURIComponent(card.dataset.code);
            const language = card.dataset.language || 'python';
            const input = card.dataset.input ? decodeURIComponent(card.dataset.input) : '{}';
            
            elements.codeEditor.value = code;
            elements.inputValues.value = input;
            
            // Update language selector
            if (elements.languageSelect) {
                elements.languageSelect.value = language;
                state.language = language;
            }
            
            updateLineNumbers();
            elements.examplesModal.classList.remove('show');
            showToast(`${language.toUpperCase()} example loaded`, 'success');
        });
    });
}

function filterExamples() {
    if (!state.examples) return;
    
    const filtered = state.exampleFilter === 'all' 
        ? state.examples 
        : state.examples.filter(ex => (ex.language || 'python') === state.exampleFilter);
    
    renderExamples(filtered);
}

// ============================================================================
// Visualization Rendering
// ============================================================================

function renderVisualization() {
    if (state.steps.length === 0) return;
    
    elements.totalSteps.textContent = state.steps.length;
    renderTimeline();
    displayStep(0);
}

function displayStep(index) {
    if (index < 0 || index >= state.steps.length) return;
    
    state.currentStepIndex = index;
    const step = state.steps[index];
    const explanation = state.explanations[index] || {};
    
    // Update progress
    const progress = ((index + 1) / state.steps.length) * 100;
    elements.progressFill.style.width = `${progress}%`;
    elements.currentStep.textContent = index + 1;
    
    // Update current code
    elements.currentCode.innerHTML = `<code class="language-python">${escapeHtml(step.code_snippet)}</code>`;
    hljs.highlightElement(elements.currentCode.querySelector('code'));
    
    // Update variables
    renderVariables(step.state_after);
    
    // Update array visualization
    renderArray(step.state_after);
    
    // Update flowchart
    renderFlowchart(step);
    
    // Update explanation
    elements.stepBadge.textContent = `Step ${index + 1}`;
    elements.stepType.textContent = formatStepType(step.step_type);
    elements.explanationSummary.textContent = explanation.summary || step.description;
    elements.explanationDetailed.textContent = explanation.detailed_explanation || '-';
    elements.explanationWhy.textContent = explanation.why_this_happens || '-';
    
    if (explanation.what_to_notice && explanation.what_to_notice.length > 0) {
        elements.explanationNotice.innerHTML = explanation.what_to_notice
            .map(n => `<li>${n}</li>`).join('');
    }
    
    if (explanation.common_mistakes && explanation.common_mistakes.length > 0) {
        elements.explanationMistakes.innerHTML = explanation.common_mistakes
            .map(m => `<li>${m}</li>`).join('');
    }
    
    elements.complexityText.textContent = explanation.complexity_note || 'O(?)';
    
    // Update timeline active
    document.querySelectorAll('.timeline-step').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });
}

function renderVariables(stateAfter) {
    if (!stateAfter || Object.keys(stateAfter).length === 0) {
        elements.variablesGrid.innerHTML = '<div class="empty-state">No variables</div>';
        return;
    }
    
    elements.variablesGrid.innerHTML = Object.entries(stateAfter)
        .map(([name, value]) => `
            <div class="var-item">
                <span class="var-name">${name}</span>
                <span class="var-value">${formatValue(value)}</span>
            </div>
        `).join('');
}

function renderArray(stateAfter) {
    if (!stateAfter) return;
    
    const arrays = Object.entries(stateAfter).filter(([_, v]) => Array.isArray(v));
    
    if (arrays.length === 0) {
        elements.arraySection.style.display = 'none';
        return;
    }
    
    elements.arraySection.style.display = 'block';
    const [name, arr] = arrays[0];
    
    elements.arrayContainer.innerHTML = arr.map((val, i) => `
        <div class="array-item" data-index="${i}">${val}</div>
    `).join('');
    
    elements.arrayIndices.innerHTML = arr.map((_, i) => `
        <span>${i}</span>
    `).join('');
}

function renderFlowchart(step) {
    const stepType = step.step_type;
    let nodeClass = '';
    
    if (stepType.includes('loop')) nodeClass = 'loop';
    else if (stepType.includes('condition')) nodeClass = 'condition';
    
    const isActive = true;
    
    elements.flowchart.innerHTML = `
        <div class="flow-node ${nodeClass} ${isActive ? 'active' : ''}">
            ${formatStepType(stepType)}: Line ${step.line_number}
        </div>
    `;
}

function renderTimeline() {
    elements.timeline.innerHTML = state.steps.map((step, i) => `
        <div class="timeline-step ${i === 0 ? 'active' : ''}" data-index="${i}">
            <div class="step-num">${i + 1}</div>
            <div class="step-desc">${truncate(step.description, 15)}</div>
        </div>
    `).join('');
    
    document.querySelectorAll('.timeline-step').forEach(el => {
        el.addEventListener('click', () => {
            const index = parseInt(el.dataset.index);
            displayStep(index);
        });
    });
}

function updateComplexityDisplay() {
    if (!state.complexity) return;
    
    elements.timeComplexity.textContent = state.complexity.time_complexity || '-';
    elements.spaceComplexity.textContent = state.complexity.space_complexity || '-';
    elements.complexityExplanation.textContent = state.complexity.explanation || '-';
}

// ============================================================================
// Playback Controls
// ============================================================================

function togglePlayback() {
    if (state.steps.length === 0) {
        showToast('Analyze code first', 'error');
        return;
    }
    
    state.isPlaying = !state.isPlaying;
    updatePlayIcon(state.isPlaying);
    
    if (state.isPlaying) {
        startPlayback();
    } else {
        stopPlayback();
    }
}

function startPlayback() {
    const interval = 1000 / state.playbackSpeed;
    
    state.playbackInterval = setInterval(() => {
        if (state.currentStepIndex < state.steps.length - 1) {
            displayStep(state.currentStepIndex + 1);
        } else {
            stopPlayback();
            state.isPlaying = false;
            updatePlayIcon(false);
        }
    }, interval);
}

function stopPlayback() {
    if (state.playbackInterval) {
        clearInterval(state.playbackInterval);
        state.playbackInterval = null;
    }
}

function stepForward() {
    if (state.currentStepIndex < state.steps.length - 1) {
        displayStep(state.currentStepIndex + 1);
    }
}

function stepBackward() {
    if (state.currentStepIndex > 0) {
        displayStep(state.currentStepIndex - 1);
    }
}

function updateSpeed() {
    state.playbackSpeed = parseFloat(elements.speedSlider.value);
    elements.speedValue.textContent = `${state.playbackSpeed}x`;
    
    if (state.isPlaying) {
        stopPlayback();
        startPlayback();
    }
}

function updatePlayIcon(isPlaying) {
    const playIcon = elements.playPauseBtn.querySelector('.play-icon');
    const pauseIcon = elements.playPauseBtn.querySelector('.pause-icon');
    
    if (playIcon && pauseIcon) {
        playIcon.style.display = isPlaying ? 'none' : 'block';
        pauseIcon.style.display = isPlaying ? 'block' : 'none';
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatValue(value) {
    if (Array.isArray(value)) {
        return `[${value.join(', ')}]`;
    }
    if (typeof value === 'object' && value !== null) {
        return JSON.stringify(value);
    }
    return String(value);
}

function formatStepType(type) {
    return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function truncate(str, len) {
    return str.length > len ? str.substring(0, len) + '...' : str;
}

function showLoading(show) {
    elements.loadingOverlay.classList.toggle('show', show);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// ============================================================================
// GAMIFICATION SYSTEM
// ============================================================================

const gamification = {
    // Cache for stats
    stats: null,
    allAchievements: null,
    
    // DOM Elements
    elements: {
        statsBtn: document.getElementById('stats-btn'),
        statsModal: document.getElementById('stats-modal'),
        closeStatsModal: document.getElementById('close-stats-modal'),
        
        // Header displays
        headerXp: document.getElementById('header-xp'),
        headerLevel: document.getElementById('header-level'),
        streakCount: document.getElementById('streak-count'),
        
        // Modal displays
        modalLevel: document.getElementById('modal-level'),
        modalXpBar: document.getElementById('modal-xp-bar'),
        modalXp: document.getElementById('modal-xp'),
        modalXpMax: document.getElementById('modal-xp-max'),
        modalXpNeeded: document.getElementById('modal-xp-needed'),
        modalRank: document.getElementById('modal-rank'),
        modalStreak: document.getElementById('modal-streak'),
        modalAnalyses: document.getElementById('modal-analyses'),
        modalRuns: document.getElementById('modal-runs'),
        modalQuestions: document.getElementById('modal-questions'),
        modalLanguages: document.getElementById('modal-languages'),
        achievementsGrid: document.getElementById('achievements-grid'),
        achievementsCount: document.getElementById('achievements-count'),
        achievementsTotal: document.getElementById('achievements-total'),
        leaderboardList: document.getElementById('leaderboard-list'),
        
        // Toasts
        xpToast: document.getElementById('xp-toast'),
        xpToastAmount: document.getElementById('xp-toast-amount'),
        achievementToast: document.getElementById('achievement-toast'),
        achievementToastIcon: document.getElementById('achievement-toast-icon'),
        achievementToastName: document.getElementById('achievement-toast-name')
    },
    
    // Initialize gamification system
    init: function() {
        this.bindEvents();
        this.loadStats();
        this.loadAllAchievements();
    },
    
    // Bind event listeners
    bindEvents: function() {
        // Stats button click
        if (this.elements.statsBtn) {
            this.elements.statsBtn.addEventListener('click', () => this.openStatsModal());
        }
        
        // Close modal
        if (this.elements.closeStatsModal) {
            this.elements.closeStatsModal.addEventListener('click', () => this.closeStatsModal());
        }
        
        // Close modal on overlay click
        const modal = this.elements.statsModal;
        if (modal) {
            modal.querySelector('.modal-overlay').addEventListener('click', () => this.closeStatsModal());
        }
    },
    
    // Get auth token
    getToken: function() {
        return localStorage.getItem('token');
    },
    
    // Load user stats
    loadStats: async function() {
        const token = this.getToken();
        if (!token) return;
        
        try {
            const response = await fetch('/api/auth/stats', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            if (response.ok) {
                this.stats = await response.json();
                this.updateHeaderDisplay();
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    },
    
    // Load all achievements
    loadAllAchievements: async function() {
        try {
            const response = await fetch('/api/auth/achievements/all');
            if (response.ok) {
                this.allAchievements = await response.json();
            }
        } catch (error) {
            console.error('Failed to load achievements:', error);
        }
    },
    
    // Update header display
    updateHeaderDisplay: function() {
        if (!this.stats) return;
        
        if (this.elements.headerXp) {
            this.elements.headerXp.textContent = this.formatXP(this.stats.xp);
        }
        if (this.elements.headerLevel) {
            this.elements.headerLevel.textContent = `Lv.${this.stats.level}`;
        }
        if (this.elements.streakCount) {
            this.elements.streakCount.textContent = this.stats.streak;
        }
    },
    
    // Format XP for display (1000 -> 1K)
    formatXP: function(xp) {
        if (xp >= 1000) {
            return (xp / 1000).toFixed(1).replace(/\.0$/, '') + 'K';
        }
        return xp;
    },
    
    // Open stats modal
    openStatsModal: async function() {
        // Refresh data
        await Promise.all([
            this.loadStats(),
            this.loadLeaderboard()
        ]);
        
        if (!this.stats) return;
        
        // Update modal displays
        const s = this.stats;
        const XP_PER_LEVEL = 100;
        
        if (this.elements.modalLevel) this.elements.modalLevel.textContent = s.level;
        if (this.elements.modalXp) this.elements.modalXp.textContent = s.xp;
        if (this.elements.modalXpMax) this.elements.modalXpMax.textContent = s.level * XP_PER_LEVEL;
        if (this.elements.modalXpNeeded) this.elements.modalXpNeeded.textContent = s.xp_to_next_level;
        if (this.elements.modalXpBar) this.elements.modalXpBar.style.width = `${s.xp_progress_percent}%`;
        if (this.elements.modalRank) this.elements.modalRank.textContent = s.rank || '?';
        if (this.elements.modalStreak) this.elements.modalStreak.textContent = s.streak;
        if (this.elements.modalAnalyses) this.elements.modalAnalyses.textContent = s.total_analyses;
        if (this.elements.modalRuns) this.elements.modalRuns.textContent = s.total_runs;
        if (this.elements.modalQuestions) this.elements.modalQuestions.textContent = s.total_questions;
        
        // Languages
        this.renderLanguages(s.languages_used);
        
        // Achievements
        this.renderAchievements(s.achievements);
        
        // Show modal
        if (this.elements.statsModal) {
            this.elements.statsModal.classList.add('show');
        }
    },
    
    // Close stats modal
    closeStatsModal: function() {
        if (this.elements.statsModal) {
            this.elements.statsModal.classList.remove('show');
        }
    },
    
    // Render languages
    renderLanguages: function(languages) {
        const container = this.elements.modalLanguages;
        if (!container) return;
        
        if (!languages || languages.length === 0) {
            container.innerHTML = '<span class="no-languages">Start coding to track languages!</span>';
            return;
        }
        
        container.innerHTML = languages.map(lang => 
            `<span class="language-tag ${lang.toLowerCase()}">${this.capitalizeFirst(lang)}</span>`
        ).join('');
    },
    
    // Render achievements
    renderAchievements: function(earnedAchievements) {
        const container = this.elements.achievementsGrid;
        if (!container || !this.allAchievements) return;
        
        const earnedIds = (earnedAchievements || []).map(a => a.id || a);
        
        if (this.elements.achievementsCount) {
            this.elements.achievementsCount.textContent = earnedIds.length;
        }
        if (this.elements.achievementsTotal) {
            this.elements.achievementsTotal.textContent = this.allAchievements.length;
        }
        
        container.innerHTML = this.allAchievements.map(achievement => {
            const isEarned = earnedIds.includes(achievement.id);
            return `
                <div class="achievement-card ${isEarned ? 'earned' : 'locked'}">
                    <span class="achievement-icon">${achievement.icon}</span>
                    <span class="achievement-name">${achievement.name}</span>
                    <span class="achievement-xp">+${achievement.xp} XP</span>
                    <div class="achievement-tooltip">${achievement.desc}</div>
                </div>
            `;
        }).join('');
    },
    
    // Load leaderboard
    loadLeaderboard: async function() {
        try {
            const response = await fetch('/api/auth/leaderboard');
            if (response.ok) {
                const leaderboard = await response.json();
                this.renderLeaderboard(leaderboard);
            }
        } catch (error) {
            console.error('Failed to load leaderboard:', error);
        }
    },
    
    // Render leaderboard
    renderLeaderboard: function(leaderboard) {
        const container = this.elements.leaderboardList;
        if (!container) return;
        
        const currentUser = JSON.parse(localStorage.getItem('user') || '{}');
        
        container.innerHTML = leaderboard.map(entry => `
            <div class="leaderboard-entry ${entry.name === currentUser.name ? 'current-user' : ''}">
                <span class="leaderboard-rank">${entry.rank}</span>
                <span class="leaderboard-name">${entry.name}</span>
                <div class="leaderboard-stats">
                    <span class="leaderboard-xp">⭐ ${entry.xp} XP</span>
                    <span class="leaderboard-level">Lv.${entry.level}</span>
                    <span class="leaderboard-streak">🔥 ${entry.streak}</span>
                </div>
            </div>
        `).join('');
    },
    
    // Award XP for an action
    awardXP: async function(action, language = null) {
        const token = this.getToken();
        if (!token) return null;
        
        try {
            const response = await fetch('/api/auth/xp/award', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action, language })
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Show XP toast
                this.showXPToast(result.xp_earned);
                
                // Check for level up
                if (result.level_up) {
                    this.showLevelUp(result.level);
                }
                
                // Show new achievements
                if (result.new_achievements && result.new_achievements.length > 0) {
                    result.new_achievements.forEach((a, i) => {
                        setTimeout(() => this.showAchievementToast(a), i * 2000);
                    });
                }
                
                // Update stats
                this.stats = null;
                this.loadStats();
                
                return result;
            }
        } catch (error) {
            console.error('Failed to award XP:', error);
        }
        return null;
    },
    
    // Show XP toast
    showXPToast: function(amount) {
        const toast = this.elements.xpToast;
        const amountEl = this.elements.xpToastAmount;
        
        if (!toast || !amountEl) return;
        
        amountEl.textContent = amount;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    },
    
    // Show level up animation
    showLevelUp: function(level) {
        const levelBadge = this.elements.headerLevel;
        if (levelBadge) {
            levelBadge.textContent = `Lv.${level}`;
            levelBadge.classList.add('level-up-animation');
            setTimeout(() => {
                levelBadge.classList.remove('level-up-animation');
            }, 1000);
        }
        
        showToast(`🎉 Level Up! You're now Level ${level}!`, 'success');
    },
    
    // Show achievement toast
    showAchievementToast: function(achievement) {
        const toast = this.elements.achievementToast;
        const iconEl = this.elements.achievementToastIcon;
        const nameEl = this.elements.achievementToastName;
        
        if (!toast || !iconEl || !nameEl) return;
        
        iconEl.textContent = achievement.icon;
        nameEl.textContent = achievement.name;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 4000);
    },
    
    // Helper function
    capitalizeFirst: function(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
};

// Initialize gamification on page load
document.addEventListener('DOMContentLoaded', () => {
    gamification.init();
});

