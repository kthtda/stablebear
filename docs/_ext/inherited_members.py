"""Keep class entries compact while exposing inherited public members."""

from sphinx.ext.autodoc import ClassDocumenter


class ExpandableClassDocumenter(ClassDocumenter):
    def document_members(self, all_members=False):
        super().document_members(all_members)
        if self.doc_as_attr or self.options.inherited_members:
            return
        if not self.object.__module__.startswith("stablebear."):
            return

        inherited = {
            name
            for base in self.object.__mro__[1:]
            for name in vars(base)
            if not name.startswith("_") and name not in vars(self.object)
        }
        if not inherited:
            return

        source = self.get_sourcename()
        target = f"{self.modname}::{'.'.join(self.objpath)}"
        for line in [
            "",
            ".. dropdown:: All members, including inherited",
            "",
            f"   .. autoclass:: {target}",
            "      :members:",
            "      :inherited-members:",
            "      :undoc-members:",
            "      :no-index:",
            "",
        ]:
            self.add_line(line, source)


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx_design")
    app.add_autodocumenter(ExpandableClassDocumenter, override=True)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
