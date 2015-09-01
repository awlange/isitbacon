$(document).ready(function() {

    $("form").submit(function(event) {
        event.preventDefault();
        var inputElement = event.currentTarget.children[0].children[1];
        console.log(event);
        var inputElementId = inputElement.getAttribute("id");
        var inputElementvalue = inputElement.value;

        console.log(inputElementId);
        // Now let's try to validate that
        var info = inputElementId.split("_");
        //$.post("/validate/field", {
        //    "eventtype": info[1],
        //    "field_key": info[2],
        //    "version": info[3],
        //    "value": inputElementvalue
        //}, function(data, status, xhr){
        //    console.log(status);
        //});

        $.ajax({
            beforeSend: function(xhrObj){
                xhrObj.setRequestHeader("Content-Type","application/json");
                xhrObj.setRequestHeader("Accept","application/json");
            },
            type: "POST",
            url: "/validate/field",
            data: JSON.stringify({
                "eventtype": info[1],
                "field_key": info[2],
                "version": info[3],
                "value": inputElementvalue
            }),
            dataType: "json",
            success: function(data, status, xhr){
                console.log(data);

                var panelType = "panel-default";
                var validation = data.label.split(":")[0];
                if (validation == "VALID") {
                    panelType = "panel-success";
                } else if (validation == "INVALID") {
                    panelType = "panel-danger";
                }

                var panelOutter = document.createElement("div");
                panelOutter.setAttribute("class", "panel " + panelType);
                var panelHeading = document.createElement("div");
                panelHeading.setAttribute("class", "panel-heading");
                panelHeading.innerHTML = data.value;
                var panelBody = document.createElement("div");
                panelBody.setAttribute("class", "panel-body");
                panelBody.innerHTML = data.label;
                panelOutter.appendChild(panelHeading);
                panelOutter.appendChild(panelBody);
                event.currentTarget.appendChild(panelOutter);
            }
        });

    });
});