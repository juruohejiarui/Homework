def divFunc(a, b) -> int:
    if int(b) == 0: raise ZeroDivisionError
    return int(a) // int(b)

if __name__ == "__main__":
    try:
        divFunc("114514", 1919)
        divFunc("114514.11", 0)
    except ZeroDivisionError:
        print("The exception \"zero division error\" triggered.")
    except Exception as exception:
        print(f"Unknown exception triggered. {exception}")
    else:
        print("Succfullly conducting all the function call")
    finally:
        print("end")

    assert 1919810 // 10 != 191981

