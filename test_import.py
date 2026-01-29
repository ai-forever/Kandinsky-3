try:
    import kandinsky3
    print("\033[32mYES\033[0m")
except Exception as e:
    print(f"\033[31mNO:\033[0m {e}")