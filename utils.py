import cohere
import altair as alt
import umap
import numpy as np


__all__ = ['generate_prompts', 'generate_topics', 'create_interactive_scatterplot']

api_key = 'NIL'
co = cohere.Client(api_key)

def generate_prompts(topic_info_df):
    prompts = []

    for i in range(1, len(topic_info_df)):
        keywords = ','.join(topic_info_df.iloc[i]['Representation'])
        sample_docs = ''

        for j, doc in enumerate(topic_info_df.iloc[i]['Representative_Docs']):
            sample_docs += "Review " + str (j + 1) + ": " + doc + '\n'

        prompt = f"""I have a set of customer reviews that need to be categorized into topics for departmental segmentation. The reviews are as follows:\n{sample_docs}\nThe topic is described by the following keywords: {keywords}\nBased on the information provided, please suggest one formal and specific 1-2 word label for the topic that is suitable for a business context."""
        prompts.append(prompt)
    
    return prompts

def generate_topics(prompts):
    generated_topics = [""]

    for prompt in prompts:
        response = co.chat(message=prompt, model="command-r-plus", temperature=0.3)
        generated_topics.append(response.text)

    return generated_topics

def sub_func_create_scatterplot(
        df,
        fields_in_tooltip = None,
        title = '',
        title_column = 'keywords'
):
    if fields_in_tooltip is None:
        fields_in_tooltip = ['']

    selection = alt.selection_multi(fields=[title_column], bind='legend')

    chart = alt.Chart(df).transform_calculate(
    ).mark_circle(size=20, stroke='#666', strokeWidth=1, opacity=0.1).encode(
        x=
        alt.X('x',
              scale=alt.Scale(zero=False),
              axis=alt.Axis(labels=False, ticks=False, domain=False)
              ),
        y=
        alt.Y('y',
              scale=alt.Scale(zero=False),
              axis=alt.Axis(labels=False, ticks=False, domain=False)
              ),

        color=alt.Color(f'{title_column}:N',
                        legend=alt.Legend(columns=2,
                                          symbolLimit=0,
                                          orient='right',
                                          labelFontSize=12)
                        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=fields_in_tooltip
    ).properties(
        width=600,
        height=400
    ).add_selection(
        selection
    ).configure_legend(labelLimit=0).configure_view(
        strokeWidth=0
    ).configure(background="#F6f6f6").properties(
        title=title
    ).configure_range(
        category={'scheme': 'category20'}
    ).properties(
        width='container'
    )
    return chart

def create_interactive_scatterplot(df):
    print("Reached in CREATE PLOT FUNCTION!")
    embeddings = co.embed(texts=list(df['Reviews']), truncate="RIGHT").embeddings
    embeddings = np.array(embeddings)
    print('FINISHED EMBEDDING!')
    n_neighbors = 15
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    umap_embeds = reducer.fit_transform(embeddings)
    df['x'] = umap_embeds[:, 0]
    df['y'] = umap_embeds[:, 1]

    print('FINISHED SETTING X,Y COORDS!')

    title_column = 'Topic_Name'
    fields_in_tooltip = ['Reviews',  'Topic_Name']

    title = "Interactive scatterplot of your data (hover over the points for more info!)"

    chart = sub_func_create_scatterplot(df,
                     fields_in_tooltip=fields_in_tooltip,
                     title=title,
                     title_column=title_column)
    print('ABOUT TO RETURN CHART!')
    return chart