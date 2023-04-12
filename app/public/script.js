async function predict() {
    
    var data = ['age',
        'self_employed_0', 'self_employed_1',
        'no_of_employees_1-5', 'no_of_employees_100-500', 'no_of_employees_26-100', 'no_of_employees_500-1000', 'no_of_employees_6-25', 'no_of_employees_>1000',
        'tech_company_0.0', 'tech_company_1.0',
        "mental_healthcare_coverage_I don't know", "mental_healthcare_coverage_No", "mental_healthcare_coverage_Yes",
        'knowledge_about_mental_healthcare_options_workplace_I am not sure', 'knowledge_about_mental_healthcare_options_workplace_No', 'knowledge_about_mental_healthcare_options_workplace_Yes',
        "employer_discussed_mental_health _I don't know", 'employer_discussed_mental_health _No', 'employer_discussed_mental_health _Yes',
        "employer_offer_resources_to_learn_about_mental_health_I don't know", 'employer_offer_resources_to_learn_about_mental_health_No', 'employer_offer_resources_to_learn_about_mental_health_Yes',
        "medical_leave_from_work _I don't know", 'medical_leave_from_work _Neither easy nor difficult', 'medical_leave_from_work _Somewhat difficult', 'medical_leave_from_work _Somewhat easy', 'medical_leave_from_work _Very difficult', 'medical_leave_from_work _Very easy',
        'comfortable_discussing_with_coworkers_Maybe', 'comfortable_discussing_with_coworkers_No', 'comfortable_discussing_with_coworkers_Yes',
        "employer_take_mental_health_seriously_I don't know", 'employer_take_mental_health_seriously_No', 'employer_take_mental_health_seriously_Yes',
        "openess_of_family_friends_I don't know", 'openess_of_family_friends_Neutral', 'openess_of_family_friends_Not open at all', 'openess_of_family_friends_Somewhat not open', 'openess_of_family_friends_Somewhat open', 'openess_of_family_friends_Very open',
        "family_history_mental_illness_I don't know", 'family_history_mental_illness_No', 'family_history_mental_illness_Yes',
        'mental_health_disorder_past_Maybe', 'mental_health_disorder_past_No', 'mental_health_disorder_past_Yes',
        'currently_mental_health_disorder_Maybe', 'currently_mental_health_disorder_No', 'currently_mental_health_disorder_Yes',
        'gender_female', 'gender_male', 'gender_other',
        'country_Afghanistan', 'country_Algeria', 'country_Argentina', 'country_Australia', 'country_Austria',
        'country_Bangladesh', 'country_Belgium', 'country_Bosnia and Herzegovina', 'country_Brazil', 'country_Brunei', 'country_Bulgaria', 'country_Canada', 'country_Chile', 'country_China', 'country_Colombia', 'country_Costa Rica', 'country_Czech Republic', 'country_Denmark', 'country_Ecuador', 'country_Estonia', 'country_Finland', 'country_France', 'country_Germany', 'country_Greece', 'country_Guatemala', 'country_Hungary', 'country_India', 'country_Iran', 'country_Ireland', 'country_Israel', 'country_Italy', 'country_Japan', 'country_Lithuania', 'country_Mexico', 'country_Netherlands', 'country_New Zealand', 'country_Norway', 'country_Other', 'country_Pakistan', 'country_Poland', 'country_Romania', 'country_Russia', 'country_Serbia', 'country_Slovakia', 'country_South Africa', 'country_Spain', 'country_Sweden', 'country_Switzerland', 'country_Taiwan', 'country_United Kingdom', 'country_United States of America', 'country_Venezuela', 'country_Vietnam',
        'country work _Afghanistan', 'country work _Argentina', 'country work _Australia', 'country work _Austria',
        'country work _Bangladesh', 'country work _Belgium', 'country work _Bosnia and Herzegovina', 'country work _Brazil', 'country work _Brunei', 'country work _Bulgaria', 'country work _Canada', 'country work _Chile', 'country work _China', 'country work _Colombia', 'country work _Costa Rica', 'country work _Czech Republic', 'country work _Denmark', 'country work _Ecuador', 'country work _Estonia', 'country work _Finland', 'country work _France', 'country work _Germany', 'country work _Greece', 'country work _Guatemala', 'country work _Hungary', 'country work _India', 'country work _Iran', 'country work _Ireland', 'country work _Israel', 'country work _Italy', 'country work _Japan', 'country work _Lithuania', 'country work _Mexico', 'country work _Netherlands', 'country work _New Zealand', 'country work _Norway', 'country work _Other', 'country work _Pakistan', 'country work _Poland', 'country work _Romania', 'country work _Russia', 'country work _Serbia', 'country work _Slovakia', 'country work _South Africa', 'country work _Spain', 'country work _Sweden', 'country work _Switzerland', 'country work _Turkey', 'country work _United Arab Emirates', 'country work _United Kingdom', 'country work _United States of America', 'country work _Venezuela', 'country work _Vietnam',
        'work_remotely_Always', 'work_remotely_Never', 'work_remotely_Sometimes', 'tech_role_0', 'tech_role_1'
    ]

    var ages = [33, 40, 21, 36, 42, 26, 29, 30, 56, 35, 51, 24, 38, 44, 27, 55, 22,
        25, 28, 23, 32, 31, 43, 37, 39, 45, 46, 20, 54, 34, 61, 41, 48, 66,
        19, 52, 50, 49, 47, 57, 74, 53, 58, 70, 59, 62, 63, 65
    ];

    /* The above code is checking if the value of an HTML input element with the ID "age" is empty. If
    it is empty, it will display an error message using the SweetAlert library. */
    if (document.getElementById('age').value == '') {
        swal({
            title: "Error!",
            text: "Please enter your age!",
            icon: "error",
          });
    }
    
    else {
        var age = parseInt(document.getElementById('age').value);
    ages.push(age);

    var scaledAges = [];
    var min = Math.min.apply(null, ages);
    var max = Math.max.apply(null, ages);
    for (var i = 0; i < ages.length; i++) {
        scaledAges.push((ages[i] - min) / (max - min));
    }
    console.log(scaledAges);

    /* The above code is processing a mental health survey form by creating an array of 164 elements,
    filling it with zeros, and then updating the array with the user's responses to the survey
    questions. The updated array is then converted to a Float64Array and used as input to an ONNX
    model loaded by the code. The model is run on the input data, and the resulting output is used
    to generate a message to display to the user based on their survey responses. */
    var myArray = new Array(164).fill(0);
    myArray[0] = scaledAges[scaledAges.length - 1];
    console.log(myArray)

    var form = document.getElementById('mental_health_survey');
    for (var i = 1; i < form.length - 1; i++) {
        let index = data.indexOf(form.elements[i].value);
        myArray[index] = 1;
    }

    const x = new Float64Array(myArray);
    console.log(x);
    const sess = new onnx.InferenceSession();
    await sess.loadModel('onnx_model.onnx');
    const input = new onnx.Tensor(x, 'float64', [1, 164]);
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const result = Math.round(outputTensor.data * 100);
    var titlemessage = getMessage(result);
    var title;
    var message;
    title = titlemessage[0];
    message = titlemessage[1];

    swal({
        title: title,
        text: message,
        icon: "success",
        buttons: ["Close", "Email results?"],
        })
        .then((willEmail) => {
            if (willEmail) {
                swal({
                    title: "Please enter your email address:",
                    content: "input",
                })
                .then((email) => {
                    if (email == null) {
                        swal("Email not sent!");
                    }
                    else {
                        // send email using the /send-email endpoint
                        fetch('/send-email', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                email: email,
                                subject: `Mental Health Survey Results taken on ${new Date().toLocaleString()}`,
                                text: message,
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log(data);
                        })
                        .catch(error => {
                            console.error(error);
                        });
                        swal({
                            title:"Email sent!",
                            icon: "success",
                        });
                    }
                });
            }
    });
    console.log(`Output tensor: ${outputTensor.data}`);
    }
}
/* The above code is using the jQuery library to initialize select2 dropdown menus for various form
fields. Select2 is a jQuery-based replacement for select boxes. It supports searching, remote data
sets, and infinite scrolling of results. The code is targeting specific form fields by their IDs and
applying the select2 function to them. */

$(document).ready(function() {
    $("#self-employed").select2();
    $("#no_employees").select2();
    $("#tech_company").select2();
    $("#mental_healthcare_coverage").select2();
    $("#knowledge_about_mental_healthcare_options_workplace").select2();
    $("#employer_discussed_mental_health").select2();
    $("#employer_offer_resources_to_learn_about_mental_health").select2();
    $("#medical_leave_from_work").select2();
    $("#comfortable_discussing_with_coworkers").select2();
    $("#employer_take_mental_health_seriously").select2();
    $("#openess_of_family_friends").select2();
    $("#family_history_mental_illness").select2();
    $("#mental_health_disorder_past").select2();
    $("#currently_mental_health_disorder").select2();
    $("#gender").select2();
    $("#country").select2();
    $("#country_work").select2();
    $("#work_remotely").select2();
    $("#tech_role").select2();
});

/**
 * The function provides personalized messages and recommendations based on the result of a mental
 * health survey.
 * @param result - The result parameter is a number that represents the percentage chance that seeking
 * professional support for mental health would be beneficial for the individual taking the survey. The
 * getMessage function uses this parameter to generate a title and message that provide guidance and
 * recommendations for the individual based on their survey results.
 * @returns An array containing a title and message based on the input result.
 */
function getMessage(result){
    var title;
    var message;
    if (result <= 20) {
        title = "Consider Prioritizing Your Mental Health";
        message = `We understand that mental health is an important and personal matter. Based on our analysis of your survey, there is an ${result}% chance that prioritizing your mental health could be beneficial. Taking steps to care for your mental well-being can have a positive impact on your overall health and happiness. We encourage you to consider seeking professional support to explore strategies for enhancing your mental well-being.`;
        return [title, message];
    }
    else if (result <= 50) {
        title = "Consider Professional Support for Your Mental Health";
        message = `Our analysis of your survey indicates that there is an ${result}% chance that seeking professional support for your mental health could be beneficial. We recommend that you consider talking to a mental health professional to further explore ways to support your well-being. It's okay to seek help when needed, and there are several resources available to support you in your mental health journey.`;
        return [title, message];
    }
    else if (result <= 70) {
        title = "Strongly Consider Professional Help for Your Mental Health";
        message = `Based on our analysis of your survey, there is an ${result}% chance that seeking professional support for your mental health would strongly benefit you. We strongly recommend that you consider reaching out to a mental health professional to discuss your concerns in depth. Prioritizing your mental health is important, and professional support can provide you with tools and strategies to improve your well-being.`;
        return [title, message];
    }
    else {
        title = "Urgently Seek Professional Help for Your Mental Health";
        message = `Our analysis of your survey results indicates that there is an ${result}% chance that seeking professional support for your mental health is very strongly recommended. We urge you to prioritize your mental health and seek professional help without delay. Your well-being matters, and talking to a mental health professional can provide you with valuable support and guidance in improving your mental health.`;
        return [title, message];
    }
}