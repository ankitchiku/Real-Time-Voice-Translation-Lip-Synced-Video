from pipeline import TranslationPipeline

if __name__ == "__main__":
    p = TranslationPipeline()
    report = p.sanity_check()
    print("Sanity check report:")
    for k, v in report.items():
        print(f"{k}: {v}")
