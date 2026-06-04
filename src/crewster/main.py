"""HPC CLI entry point"""

import typer

app = typer.Typer(help="HPC job execution support tool")


def _generate_skill_reference(cli_app: typer.Typer) -> str:
    """Generate markdown CLI reference from Typer app metadata for use in SKILL.md."""
    # typer 0.26+ vendored click as `typer._click`; use Typer's wrapper classes so
    # this stays correct without depending on the external `click` package.
    from typer.core import TyperArgument, TyperGroup, TyperOption

    typer_group = typer.main.get_command(cli_app)
    assert isinstance(typer_group, TyperGroup)
    lines = ["## Commands", ""]
    for name, cmd in sorted(typer_group.commands.items()):
        lines.append(f"### `crewster {name}`")
        if cmd.help:
            lines.append(cmd.help)
        params = [
            p for p in cmd.params if not isinstance(p, TyperArgument) or p.required
        ]
        args = [p for p in params if isinstance(p, TyperArgument)]
        opts = [p for p in params if isinstance(p, TyperOption) and p.name != "help"]
        if args:
            for arg in args:
                type_name = arg.type.name.upper() if arg.type else ""
                required = " (required)" if arg.required else ""
                lines.append(
                    f"- Argument: `{arg.human_readable_name}` {type_name}{required}"
                )
        if opts:
            for opt in opts:
                decls = ", ".join(f"`{d}`" for d in opt.opts)
                help_text = f": {opt.help}" if opt.help else ""
                lines.append(f"- {decls}{help_text}")
        lines.append("")

    # Generate config reference from Pydantic models
    from pydantic_core import PydanticUndefined

    from .config import HpcConfig

    lines.append("## Configuration (crewster.toml)")
    lines.append("")
    for field_name, field_info in HpcConfig.model_fields.items():
        annotation = field_info.annotation
        doc = (
            annotation.__doc__ if annotation and hasattr(annotation, "__doc__") else ""
        )
        lines.append(f"### `[{field_name}]`")
        if doc:
            lines.append(doc)
        if annotation is not None and hasattr(annotation, "model_fields"):
            for sub_name, sub_info in annotation.model_fields.items():  # type: ignore[union-attr]
                raw = str(sub_info.annotation).replace("typing.", "")
                # str(str) gives "<class 'str'>", normalize to "str"
                if raw.startswith("<class '"):
                    type_hint = raw[8:-2]
                else:
                    type_hint = raw
                if sub_info.default is PydanticUndefined:
                    lines.append(f"- `{sub_name}`: {type_hint} (required)")
                else:
                    lines.append(
                        f"- `{sub_name}`: {type_hint} (default: {sub_info.default!r})"
                    )
        lines.append("")

    return "\n".join(lines)


def _skill_callback(value: bool):
    """Print CLI reference for SKILL.md and exit."""
    if not value:
        return
    from . import cli  # noqa: F401 - register commands

    print(_generate_skill_reference(app))
    raise typer.Exit()


def _version_callback(value: bool):
    if not value:
        return
    from importlib import metadata

    print(f"crewster {metadata.version('crewster')}")
    raise typer.Exit()


@app.callback()
def app_callback(
    skill: bool = typer.Option(
        False,
        "--skill",
        hidden=True,
        callback=_skill_callback,
        is_eager=True,
        help="Print CLI reference for SKILL.md",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit",
    ),
):
    pass


def main():
    from . import cli  # noqa: F401 - register commands

    app()


if __name__ == "__main__":
    main()
