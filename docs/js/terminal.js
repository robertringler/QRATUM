// QuASIM Technical Portal - Terminal Effects

class TerminalTyping {
  constructor(element, lines, options = {}) {
    this.element = element;
    this.lines = lines;
    this.options = {
      typeSpeed: options.typeSpeed || 50,
      deleteSpeed: options.deleteSpeed || 30,
      pauseDelay: options.pauseDelay || 1000,
      loop: options.loop !== false,
      cursor: options.cursor !== false
    };
    this.currentLine = 0;
    this.currentChar = 0;
    this.isDeleting = false;
    
    if (this.options.cursor) {
      const cursor = document.createElement('span');
      cursor.className = 'terminal-cursor';
      this.element.appendChild(cursor);
    }
    
    this.type();
  }
  
  type() {
    const line = this.lines[this.currentLine];
    
    if (this.isDeleting) {
      this.element.textContent = line.substring(0, this.currentChar - 1);
      this.currentChar--;
    } else {
      this.element.textContent = line.substring(0, this.currentChar + 1);
      this.currentChar++;
    }
    
    let speed = this.isDeleting ? this.options.deleteSpeed : this.options.typeSpeed;
    
    if (!this.isDeleting && this.currentChar === line.length) {
      speed = this.options.pauseDelay;
      this.isDeleting = true;
    } else if (this.isDeleting && this.currentChar === 0) {
      this.isDeleting = false;
      this.currentLine = (this.currentLine + 1) % this.lines.length;
      speed = 500;
      
      if (!this.options.loop && this.currentLine === 0) {
        return;
      }
    }
    
    setTimeout(() => this.type(), speed);
  }
}

class AutoRunTerminal {
  constructor(element, commands) {
    this.element = element;
    this.commands = commands;
    this.currentCommand = 0;
    
    this.run();
  }
  
  run() {
    if (this.currentCommand >= this.commands.length) {
      return;
    }
    
    const cmd = this.commands[this.currentCommand];
    const line = document.createElement('div');
    line.className = 'terminal-line';
    
    const prompt = document.createElement('span');
    prompt.className = 'terminal-prompt';
    prompt.textContent = '$ ';
    
    const command = document.createElement('span');
    command.className = 'terminal-command';
    
    line.appendChild(prompt);
    line.appendChild(command);
    this.element.appendChild(line);
    
    // Type command
    this.typeText(command, cmd.command, () => {
      if (cmd.output) {
        setTimeout(() => {
          const output = document.createElement('div');
          output.className = 'terminal-output';
          output.textContent = cmd.output;
          this.element.appendChild(output);
          
          this.currentCommand++;
          setTimeout(() => this.run(), cmd.delay || 1000);
        }, 500);
      } else {
        this.currentCommand++;
        setTimeout(() => this.run(), cmd.delay || 1000);
      }
    });
  }
  
  typeText(element, text, callback, charIndex = 0) {
    if (charIndex < text.length) {
      element.textContent += text[charIndex];
      setTimeout(() => this.typeText(element, text, callback, charIndex + 1), 50);
    } else {
      callback();
    }
  }
}

// Auto-initialize terminals
document.addEventListener('DOMContentLoaded', () => {
  // Hero terminal demo
  const heroTerminal = document.getElementById('hero-terminal-body');
  if (heroTerminal) {
    new AutoRunTerminal(heroTerminal, [
      { command: 'quasim init --qubits 32 --backend cuQuantum', output: '✓ Initialized 32-qubit simulation (2.4ms)', delay: 1500 },
      { command: 'quasim circuit load quantum_fourier_transform.qasm', output: '✓ Loaded QFT circuit with 45 gates', delay: 1500 },
      { command: 'quasim run --precision fp32 --gpu 0-3', output: '✓ Execution complete: 125ms (4x GPU acceleration)', delay: 2000 },
      { command: 'quasim measure --shots 1024', output: '✓ Measurement complete: Fidelity 99.95%', delay: 1500 }
    ]);
  }
});
