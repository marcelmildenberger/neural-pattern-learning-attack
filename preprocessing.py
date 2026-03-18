from __future__ import annotations
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.containers import Center, Horizontal, Container
from textual.widgets import Label, Button, Footer, Static, Header, DataTable, RadioSet, RadioButton, Rule

from textual_fspicker import FileOpen, FileSave, Filters

import numpy as np
import pandas as pd


class ExitScreen(ModalScreen):
    """A modal exit screen."""

    DEFAULT_CSS = """
    ExitScreen {
        align: center middle;
    }

    ExitScreen > Container {
        width: auto;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    ExitScreen > Container > Label {
        width: 100%;
        content-align-horizontal: center;
        margin-top: 1;
    }

    ExitScreen > Container > Horizontal {
        width: auto;
        height: auto;
    }

    ExitScreen > Container > Horizontal > Button {
        margin: 2 4;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Are you sure you want to quit?")
            with Horizontal():
                yield Button("No", id="no", variant="error")
                yield Button("Yes", id="yes", variant="success")

    @on(Button.Pressed, "#yes")
    def exit_app(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#no")
    def back_to_app(self) -> None:
        self.app.pop_screen()


class ErrorScreen(ModalScreen):
    """A modal error screen."""

    DEFAULT_CSS = """
    ErrorScreen {
        align: center middle;
    }

    ErrorScreen > Container {
        width: auto;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    ErrorScreen > Container > Label {
        width: 100%;
        content-align-horizontal: center;
        margin-top: 1;
    }

    ErrorScreen > Container > Horizontal {
        width: auto;
        height: auto;
    }

    ErrorScreen > Container > Horizontal > Button {
        margin: 2 4;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self.app.msg)
            with Horizontal():
                yield Button("OK", id="ok", variant="error")

    @on(Button.Pressed, "#ok")
    def back_to_app(self) -> None:
        self.app.pop_screen()


class StartupScreen(Screen):
    CSS = """
    StartupScreen {
        align: center middle;
    }

    StartupScreen Horizontal {
        align: center middle;
        height: auto;
        margin-bottom: 1;
    }

    StartupScreen Horizontal Button {
        margin-left: 1;
        margin-right: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            yield Label("Select the file containing your data.")
        with Horizontal():
            yield Button("Open data file", id="open", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#open")
    def open_file(self) -> None:
        """Show the `FileOpen` dialog when the button is pushed."""
        self.app.push_screen(
            FileOpen(
                ".",
                filters=Filters(
                    ("CSV", lambda p: p.suffix.lower() == ".csv"),
                    ("XLS", lambda p: p.suffix.lower() == ".xls"),
                    ("XLSX", lambda p: p.suffix.lower() == ".xlsx"),
                ),
            ),
            callback=self.process_selection,
        )

    def process_selection(self, to_show: Path | None) -> None:
        """Show the file that was selected by the user.

        Args:
            to_show: The file to show.
        """
        if to_show is None:
            self.query_one(Label).update("Cancelled")
        else:
            self.app.open_path = str(to_show)
            ftype = str(to_show).split(".")[-1]
            if ftype.lower() == "csv":
                self.app.loaded_df = pd.read_csv(self.app.open_path)
            else:
                self.app.loaded_df = pd.read_excel(self.app.open_path)
            self.app.data_rows = self.app.loaded_df.values.tolist()
            self.app.data_cols = self.app.loaded_df.columns.tolist()
            # self.app.uid_col = data.columns.tolist()[0]
            # self.app.selected_data_cols = data.columns.tolist()[1:]
            self.app.push_screen(ColSelectorScreen())


class ColSelectorScreen(Screen):
    CSS = """
    ColSelectorScreen {
        align: center middle;
    }

    DataTable {
    height: 25;
    min-width: 150;
    max-width: 250;
    }

    RadioSet {
    min-width: 15;
    max-width: 25;    
    }

    Button{
        width: 16;
    }

    #save {
        dock: right;
    }


    Screen{

    }

    #instruction{
        align: center top;
        text-align: center;
        height: 8
    }

    """

    instr_text = """

    Please select the column containing the unique ID of the individuals using the buttons on the left. If your data does not contain a unique ID, the program will automatically create a new one.

    Afterwards, you may select the attributes to be used for linking by clicking on the name of the column. Note that the order in which you select the columns will be the order in which they are used for linking.
    Clicking on a column name that is already part of the selection will remove it.
    The ID column cannot be part of the linking data.
    """

    def on_mount(self) -> None:

        table = self.query_one(DataTable)
        table.cursor_type = "cell"
        table.zebra_stripes = True
        table.add_columns(*self.app.data_cols)
        table.add_rows(self.app.data_rows)

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            yield Static(self.instr_text, id="instruction")
            yield Rule(orientation="horizontal", line_style="double")
            with Horizontal():
                with RadioSet(name="id_selector", id="id_selector"):
                    yield RadioButton(label="--NO ID--", name="newID", id="newID")
                    for colname in self.app.data_cols:
                        yield RadioButton(label=colname, name=colname, id=colname)
                yield Rule(orientation="vertical", line_style="double")
                yield DataTable()
                # yield Rule(orientation="vertical", line_style="double")
                # with Vertical():
                #    yield Checkbox("Create Header", id="make_header")
                #    yield Checkbox("Create ID", id="make_id")
            yield Rule(orientation="horizontal", line_style="double")
            with Horizontal():
                yield Label("Please select the ID and at least one data column.", id="selectedLabel")
                yield Button("Save", id="save", variant="success")

        yield Footer()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if str(event.pressed.label) not in self.app.selected_data_cols:
            self.app.uid_col = event.pressed.label
        else:
            self.app.msg = "The ID column cannot be part of the matching data."
            self.app.push_screen(ErrorScreen())
            event.radio_set.action_toggle_button()
            self.app.uid_col = ""
        data_str = ", ".join(self.app.selected_data_cols)
        info_str = "ID column: %s \nData columns: %s" % (self.app.uid_col, data_str)
        self.query_one("#selectedLabel", Label).update(info_str)

    def on_data_table_header_selected(self, event) -> None:
        selected_col = str(event.label)
        if selected_col != str(self.app.uid_col):
            if selected_col not in self.app.selected_data_cols:
                self.app.selected_data_cols.append(selected_col)
            else:
                self.app.selected_data_cols = [c for c in self.app.selected_data_cols if c != selected_col]

            data_str = ", ".join(self.app.selected_data_cols)
            info_str = "ID column: %s \nData columns: %s" % (self.app.uid_col, data_str)
            self.query_one("#selectedLabel", Label).update(info_str)
        else:
            self.app.msg = "The ID column cannot be part of the matching data."
            self.app.push_screen(ErrorScreen())

    @on(Button.Pressed, "#save")
    def save_file(self) -> None:
        """Show the `FileSave` dialog when the button is pushed."""
        self.app.push_screen(FileSave(can_overwrite=False,
                                      filters=Filters(
                                          ("TSV", lambda p: p.suffix.lower() == ".tsv"),
                                      ),
                                      ), callback=self.process_save)

    def process_save(self, save_path: Path | None) -> None:
        """Show the file that was selected by the user.

        Args:
            to_show: The file to show.
        """
        if save_path is not None:
            save_path = str(save_path)
            if save_path[-4:] != ".tsv":
                save_path += ".tsv"

            data = self.app.loaded_df.copy()
            data.replace(np.nan, "", inplace=True)

            if str(self.app.uid_col) != "--NO ID--":
                tmp_id_col = data[str(self.app.uid_col)]
            else:
                tmp_id_col = list(range(data.shape[0]))

            data = data[list(self.app.selected_data_cols)]
            data["uid"] = tmp_id_col
            data.to_csv(save_path, index=False, sep="\t")


class PreproApp(App[None]):
    """A simple test application."""

    TITLE = "Data Preprocessing for GMA against PPRL"

    CSS = """
    Screen#_default {
        align: center middle;
    }

    Screen#_default Horizontal {
        align: center middle;
        height: auto;
        margin-bottom: 1;
    }

    Screen#_default Horizontal Button {
        margin-left: 1;
        margin-right: 1;
    }
    """
    BINDINGS = [("d", "toggle_dark", "Light/Dark"), ("q", "quit_app", "Quit")]

    open_path = reactive("")
    data_rows = reactive(list)
    data_cols = reactive(list)

    msg = ""
    loaded_df = None
    selected_data_cols = reactive(list)
    uid_col = reactive("")

    def compose(self) -> ComposeResult:
        """Compose the layout of the test application."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(StartupScreen())

    def action_quit_app(self) -> None:
        self.push_screen(ExitScreen())


##############################################################################
if __name__ == "__main__":
    PreproApp().run()
