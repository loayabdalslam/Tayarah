from tayarah_terminal import TayarahTerminal, SymbLLMTerminalApp if "SymbLLMTerminalApp" in locals() else TayarahTerminal

# Alias for backwards compatibility
SymbLLMTerminalApp = TayarahTerminal

if __name__ == "__main__":
    app = TayarahTerminal()
    app.run()
