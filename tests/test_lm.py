from factscore.lm import LM


class EchoLM(LM):
    def load_model(self):
        self.model = object()

    def _generate(self, prompt, **kwargs):
        if isinstance(prompt, list):
            return [(value, None) for value in prompt]
        return prompt, None


def test_lm_supports_disabled_cache():
    model = EchoLM(cache_file=None)

    assert model.generate("hello") == ("hello", None)
    model.save_cache()
